"""Triagem segura de imagens. Inference feita pelo servidor privado no Colab.

A imagem original NUNCA e enviada para a API: apenas JPEG convertido em memoria.
"""
from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import threading
import time
import uuid
from typing import Callable

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

EXTENSOES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
CAMPOS = ['data_hora', 'execucao', 'status', 'pasta', 'origem', 'destino',
          'confianca_max', 'num_deteccoes', 'classes', 'deteccoes_json', 'erro',
          'modelo', 'confianca_minima', 'confianca_alta', 'simulacao']
PASTAS = ('deteccao', 'duvida', 'sem_deteccoes')


@dataclass(frozen=True)
class Config:
    origem: Path
    destino: Path
    url: str
    chave: str
    modelo: str = 'yolo26'
    classes: str = '0'
    prompts: str = ''
    conf_min: float = 0.12
    conf_alta: float = 0.45
    lote: int = 4
    simular: bool = True
    recursivo: bool = True
    limite: int = 0  # 0 = todas

    def validar(self):
        src, out = self.origem.expanduser().resolve(), self.destino.expanduser().resolve()
        if not src.is_dir():
            raise ValueError(f'Pasta de origem nao encontrada: {src}')
        if src == out:
            raise ValueError('Selecione uma pasta de destino DIFERENTE da origem.')
        if src.is_relative_to(out):
            raise ValueError('A pasta de destino nao pode conter a pasta de origem.')
        if self.modelo not in ('yolo26', 'yoloe26'):
            raise ValueError('Modelo invalido.')
        if not 0.01 <= self.conf_min < self.conf_alta <= 1.0:
            raise ValueError('Use 0.01 <= confianca minima < confianca alta <= 1.')
        if not 1 <= self.lote <= 8:
            raise ValueError('Lote deve estar entre 1 e 8.')
        if self.limite < 0:
            raise ValueError('O limite nao pode ser negativo.')
        url = self.url.strip().rstrip('/')
        if not url.startswith('https://') and not url.startswith('http://127.0.0.1:'):
            raise ValueError('Use a URL HTTPS do ngrok (ou localhost para testes).')
        if not self.chave.strip():
            raise ValueError('Informe X-API-Key fornecida pelo Colab.')
        if self.modelo == 'yolo26':
            try:
                ids = [int(x.strip()) for x in self.classes.split(',') if x.strip()]
            except ValueError as exc:
                raise ValueError('IDs das classes devem ser numeros, ex.: 0,2,7.') from exc
            if not ids or min(ids) < 0:
                raise ValueError('Informe pelo menos um ID de classe.')
        else:
            prompts = [x.strip() for x in self.prompts.split(',') if x.strip()]
            if not 1 <= len(prompts) <= 20:
                raise ValueError('YOLOE requer entre 1 e 20 nomes de classes em ingles.')


def descobrir_imagens(config: Config) -> list[Path]:
    origem = config.origem.resolve()
    destino = config.destino.resolve()
    iterator = origem.rglob('*') if config.recursivo else origem.iterdir()
    paths = [p for p in iterator if p.is_file() and p.suffix.lower() in EXTENSOES
             and not p.resolve().is_relative_to(destino)]
    paths.sort(key=lambda p: str(p.relative_to(origem)).casefold())
    return paths[:config.limite] if config.limite else paths


def classe_imagem(deteccoes: list[dict], conf_min: float, conf_alta: float):
    """Sem deteccoes = sem nenhuma deteccao da CLASSE ESCOLHIDA >= conf_min."""
    confiancas = [float(d['confidence']) for d in deteccoes]
    confiancas = [x for x in confiancas if x >= conf_min]
    if not confiancas:
        return 'sem_deteccoes', 0.0
    top = max(confiancas)
    return ('deteccao' if top >= conf_alta else 'duvida'), top


def jpeg_em_memoria(path: Path) -> bytes:
    """Gera JPEG compativel com a API; mantem o original intacto no disco."""
    try:
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw)
            if img.mode == 'RGBA' or (img.mode == 'P' and 'transparency' in img.info):
                rgba = img.convert('RGBA')
                fondo = Image.new('RGB', rgba.size, (255, 255, 255))
                fondo.paste(rgba, mask=rgba.getchannel('A'))
                img = fondo
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            # Nenhum redimensionamento para imagens normais; necessario para originais enormes.
            if max(img.size) > 4096:
                img.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
            for qualidade in (90, 80, 68, 55):
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=qualidade, optimize=True)
                if buf.tell() <= 11 * 1024 * 1024:
                    return buf.getvalue()
            raise ValueError('Imagem enorme: JPEG temporario ainda excede 11 MiB.')
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f'Nao foi possivel decodificar {path.name}: {exc}') from exc


def nome_livre(pasta: Path, nome: str) -> Path:
    candidato = pasta / nome
    if not candidato.exists():
        return candidato
    stem, suffix = Path(nome).stem, Path(nome).suffix
    for index in range(2, 100_000):
        candidato = pasta / f'{stem}__{index}{suffix}'
        if not candidato.exists():
            return candidato
    raise FileExistsError(f'Nao ha nome de arquivo disponivel para {nome}')


def checksum(arquivo: Path) -> str:
    h = hashlib.sha256()
    with arquivo.open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def mover_com_verificacao(origem: Path, destino: Path):
    """Mesmo volume: rename atomico. Volumes diferentes: copia e verifica hash antes de apagar."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        raise FileExistsError(destino)
    try:
        os.rename(origem, destino)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    temporario = destino.with_name(destino.name + '.' + uuid.uuid4().hex + '.part')
    try:
        shutil.copy2(origem, temporario)
        if origem.stat().st_size != temporario.stat().st_size or checksum(origem) != checksum(temporario):
            raise IOError('Falha de integridade na copia: imagem preservada na origem.')
        if destino.exists():
            raise FileExistsError(destino)
        os.rename(temporario, destino)
        # Verificacao antes de excluir original (protege contra arquivo alterado durante copia)
        if checksum(origem) != checksum(destino):
            raise IOError('Original foi modificado durante copia: mantido na origem.')
        origem.unlink()
    finally:
        temporario.unlink(missing_ok=True)


class ErroServidor(RuntimeError):
    pass


class APIColab:
    def __init__(self, config: Config):
        self.config = config
        self.sessao = requests.Session()
        self.id_config = ''
        self.max_imagens = 8
        self.url = config.url.strip().rstrip('/')
        self.headers = {'X-API-Key': config.chave.strip(), 'ngrok-skip-browser-warning': 'true'}

    def _requisicao(self, endpoint: str, data=None, files=None, metodo='POST') -> dict:
        exc_final = None
        for tentativa in range(3):
            try:
                if metodo == 'GET':
                    resp = self.sessao.get(self.url+endpoint, headers=self.headers, timeout=(12, 60))
                else:
                    # Build multipart anew from bytes, safe across retries.
                    resp = self.sessao.post(self.url+endpoint, headers=self.headers, data=data,
                                            files=files, timeout=(20, 240))
                if resp.status_code in (429, 502, 503, 504) and tentativa < 2:
                    time.sleep(1.5 * (tentativa + 1))
                    continue
                if resp.status_code != 200:
                    try:
                        detalhe = resp.json().get('detail', resp.text[:400])
                    except ValueError:
                        detalhe = resp.text[:400]
                    raise ErroServidor(f'HTTP {resp.status_code} {endpoint}: {detalhe}')
                try:
                    result = resp.json()
                except ValueError as exc:
                    raise ErroServidor('Resposta nao e JSON (confira URL do ngrok).') from exc
                if not isinstance(result, dict):
                    raise ErroServidor('Resposta inesperada da API.')
                return result
            except (requests.RequestException, OSError) as exc:
                exc_final = exc
                if tentativa < 2:
                    time.sleep(1.5 * (tentativa + 1))
                else:
                    raise ErroServidor(f'Falha de conexao com Colab: {exc}') from exc
        raise ErroServidor(str(exc_final))

    def iniciar(self) -> dict:
        health = self._requisicao('/health', metodo='GET')
        self.max_imagens = min(8, int(health.get('max_images', 8)))
        dados = {'model_type': self.config.modelo}
        if self.config.modelo == 'yoloe26':
            dados['prompts'] = json.dumps([x.strip() for x in self.config.prompts.split(',') if x.strip()])
        info = self._requisicao('/prepare', data=dados)
        self.id_config = str(info.get('config_id', ''))
        if not self.id_config:
            raise ErroServidor('API antiga: nao informou config_id. Use o notebook v7 deste pacote.')
        return info

    def predizer(self, imagens: list[tuple[Path, bytes]]) -> list[dict]:
        dados = {'config_id': self.id_config, 'conf': str(self.config.conf_min)}
        if self.config.modelo == 'yolo26':
            dados['classes'] = self.config.classes
        files = [('files', (p.name, image, 'image/jpeg')) for p, image in imagens]
        resultado = self._requisicao('/predict-batch', data=dados, files=files)
        resultados = resultado.get('results')
        if not isinstance(resultados, list) or len(resultados) != len(imagens):
            raise ErroServidor('API retornou quantidade inesperada de resultados.')
        indices = [r.get('index') for r in resultados if isinstance(r, dict)]
        if indices != list(range(len(imagens))):
            raise ErroServidor('API retornou resultados fora de ordem.')
        return resultados


def processar(config: Config, registrar: Callable[[str], None] = print,
              progresso: Callable[[int, int], None] | None = None,
              cancelar: threading.Event | None = None, api=None) -> dict:
    config.validar()
    cancelar = cancelar or threading.Event()
    origem = config.origem.resolve()
    destino = config.destino.resolve()
    imagens = descobrir_imagens(config)
    execucao = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:5]
    destino.mkdir(parents=True, exist_ok=True)
    pasta_relatorios = destino / 'relatorios'
    pasta_relatorios.mkdir(parents=True, exist_ok=True)
    caminho_csv = pasta_relatorios / f'{"simulacao_" if config.simular else "triagem_"}{execucao}.csv'
    contador = {name: 0 for name in PASTAS}
    contador.update(erro=0, cancelado=0)
    realizados = 0
    registrar(f'{len(imagens)} imagens encontradas. Modo: {"SIMULACAO (nao move)" if config.simular else "MOVER ORIGINAIS"}.')
    if not imagens:
        return {'total': 0, 'processadas': 0, 'contagem': contador, 'csv': str(caminho_csv)}
    api = api or APIColab(config)
    registrar('Conectando ao Colab e preparando modelo...')
    info = api.iniciar()
    registrar(f'Modelo pronto: {info.get("model", config.modelo)} | dispositivo: {info.get("device", "?")}')

    # Em execucao real, evita dois processos gravando na mesma pasta em paralelo.
    trava = destino / '.triagem_em_execucao.lock'
    lock_fd = None
    if not config.simular:
        try:
            lock_fd = os.open(trava, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, execucao.encode())
        except FileExistsError as exc:
            raise RuntimeError(f'Outra triagem esta usando o destino (ou travou): {trava}') from exc
    try:
        with caminho_csv.open('w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CAMPOS)
            writer.writeheader()
            def salvar(path, pasta='', resposta=None, erro='', status='ok', dest=''):
                nonlocal realizados
                resposta = resposta or {}
                dets = resposta.get('detections', [])
                classes = sorted(set(str(d.get('name', '')) for d in dets))
                linha = dict(data_hora=datetime.now().isoformat(timespec='seconds'), execucao=execucao,
                             status=status, pasta=pasta, origem=str(path), destino=str(dest),
                             confianca_max=round(max((float(d.get('confidence', 0)) for d in dets), default=0), 5),
                             num_deteccoes=len(dets), classes=', '.join(classes),
                             deteccoes_json=json.dumps(dets, ensure_ascii=False), erro=erro,
                             modelo=config.modelo, confianca_minima=config.conf_min,
                             confianca_alta=config.conf_alta, simulacao=config.simular)
                writer.writerow(linha)
                csv_file.flush()
                os.fsync(csv_file.fileno())
                realizados += 1
                if progresso:
                    progresso(realizados, len(imagens))
            lote = min(config.lote, api.max_imagens)
            for start in range(0, len(imagens), lote):
                if cancelar.is_set():
                    break
                caminhos = imagens[start:start+lote]
                validas = []
                estados = {}
                for p in caminhos:
                    try:
                        original_stat = p.stat()
                        dados = jpeg_em_memoria(p)
                        validas.append((p, dados))
                        estados[p] = (original_stat.st_size, original_stat.st_mtime_ns)
                    except Exception as exc:
                        salvar(p, erro=str(exc), status='erro')
                        contador['erro'] += 1
                        registrar(f'[ERRO LEITURA] {p.name}: {exc}')
                if not validas:
                    continue
                try:
                    respostas = api.predizer(validas)
                except ErroServidor as exc:
                    # Falha de transporte/autenticacao: PARA; arquivos pendentes permanecem na origem.
                    registrar(f'[PARADO] API indisponivel: {exc}. Nenhuma imagem pendente foi movida.')
                    raise ErroServidor(f'{exc} (processadas {realizados}/{len(imagens)}; CSV: {caminho_csv})') from exc
                for (p, _), resposta in zip(validas, respostas):
                    if cancelar.is_set():
                        break
                    if resposta.get('status') != 'ok':
                        erro = str(resposta.get('error', 'Resposta sem status ok'))
                        salvar(p, resposta=resposta, erro=erro, status='erro')
                        contador['erro'] += 1
                        registrar(f'[ERRO INFERENCIA] {p.name}: {erro}')
                        continue
                    dets = resposta.get('detections')
                    if not isinstance(dets, list):
                        salvar(p, erro='API sem lista detections', status='erro')
                        contador['erro'] += 1
                        continue
                    try:
                        pasta, confianca = classe_imagem(dets, config.conf_min, config.conf_alta)
                        rel = p.relative_to(origem)
                        pasta_dest = destino / pasta / rel.parent
                        novo = nome_livre(pasta_dest, p.name)
                        if not config.simular:
                            # Imagem alterada desde o upload? Impede mover versao diferente.
                            stat_atual = p.stat()
                            if (stat_atual.st_size, stat_atual.st_mtime_ns) != estados[p]:
                                raise RuntimeError('Arquivo original foi alterado durante inferencia; mantido na origem.')
                            mover_com_verificacao(p, novo)
                        salvar(p, pasta=pasta, resposta=resposta, dest=novo,
                               status='simulado' if config.simular else 'movido')
                        contador[pasta] += 1
                        registrar(f'[{pasta.upper()}] {rel} (conf {confianca:.1%})')
                    except Exception as exc:
                        salvar(p, resposta=resposta, erro=str(exc), status='erro')
                        contador['erro'] += 1
                        registrar(f'[ERRO MOVE] {p.name}: {exc}')
            contador['cancelado'] = len(imagens) - realizados if cancelar.is_set() else 0
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            trava.unlink(missing_ok=True)
    resumo = dict(total=len(imagens), processadas=realizados, contagem=contador,
                  csv=str(caminho_csv), origem=str(origem), destino=str(destino),
                  modelo=config.modelo, execucao=execucao, simulacao=config.simular)
    registrar('Finalizado: ' + json.dumps(contador, ensure_ascii=False))
    return resumo
