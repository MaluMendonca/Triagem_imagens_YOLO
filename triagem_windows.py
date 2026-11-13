"""Interface grafica Windows para organizar imagens com YOLO26 / YOLOE26 (Colab)."""
from __future__ import annotations
import json
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from triagem_core import APIColab, Config, ErroServidor, processar


class Aplicativo(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Triagem de Imagens com IA | YOLO26 + YOLOE-26 - v7')
        self.geometry('890x750')
        self.minsize(740, 680)
        self.eventos = queue.Queue()
        self.cancelamento = threading.Event()
        self.executando = False
        self.origem = tk.StringVar()
        self.destino = tk.StringVar()
        self.url = tk.StringVar()
        self.chave = tk.StringVar()
        self.modelo = tk.StringVar(value='yolo26')
        self.classes = tk.StringVar(value='0')
        self.prompts = tk.StringVar(value='person, forklift, tractor, truck')
        self.conf_min = tk.StringVar(value='0.12')
        self.conf_alta = tk.StringVar(value='0.45')
        self.lote = tk.StringVar(value='4')
        self.limite = tk.StringVar(value='25')
        self.recursivo = tk.BooleanVar(value=True)
        self.simular = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='Pronto. Primeiro teste em simulacao (25 imagens).')
        self.progress = tk.DoubleVar(value=0)
        self.criar_layout()
        self.after(120, self.ler_eventos)

    def criar_layout(self):
        topo = ttk.Frame(self, padding=14)
        topo.pack(fill='both', expand=True)
        ttk.Label(topo, text='TRIAGEM AUTOMÁTICA DE IMAGENS', font=('Segoe UI', 16, 'bold')).pack(anchor='w')
        ttk.Label(topo, text='YOLO26x (COCO) ou YOLOE-26m (classes por texto) • classificação e MOVIMENTAÇÃO dos originais').pack(anchor='w', pady=(2, 12))
        grid = ttk.Frame(topo)
        grid.pack(fill='x')
        grid.columnconfigure(1, weight=1)
        def linha_pasta(row, titulo, variavel):
            ttk.Label(grid, text=titulo).grid(row=row, column=0, sticky='w', pady=4)
            ttk.Entry(grid, textvariable=variavel).grid(row=row, column=1, sticky='ew', padx=8, pady=4)
            ttk.Button(grid, text='Escolher...', command=lambda: self.escolher_pasta(variavel)).grid(row=row, column=2, pady=4)
        linha_pasta(0, 'Imagens de origem', self.origem)
        linha_pasta(1, 'Pasta de saída', self.destino)
        ttk.Label(grid, text='URL HTTPS do ngrok').grid(row=2,column=0,sticky='w',pady=4)
        ttk.Entry(grid,textvariable=self.url).grid(row=2,column=1,columnspan=2,sticky='ew',padx=(8,0),pady=4)
        ttk.Label(grid,text='X-API-Key').grid(row=3,column=0,sticky='w',pady=4)
        ttk.Entry(grid,textvariable=self.chave,show='•').grid(row=3,column=1,columnspan=2,sticky='ew',padx=(8,0),pady=4)
        ttk.Separator(topo).pack(fill='x', pady=12)
        op = ttk.Frame(topo)
        op.pack(fill='x')
        self.modelo_op = op
        ttk.Label(op,text='Modelo:').pack(side='left')
        ttk.Radiobutton(op,text='YOLO26x — classes COCO',variable=self.modelo,value='yolo26', command=self.alternar_modelo).pack(side='left',padx=14)
        ttk.Radiobutton(op,text='YOLOE-26m — texto livre',variable=self.modelo,value='yoloe26',command=self.alternar_modelo).pack(side='left',padx=14)
        self.frame_classes = ttk.Frame(topo)
        self.frame_classes.pack(fill='x',pady=9)
        ttk.Label(self.frame_classes,text='IDs das classes COCO').pack(side='left')
        ttk.Entry(self.frame_classes,textvariable=self.classes,width=38).pack(side='left',padx=10,fill='x',expand=True)
        ttk.Button(self.frame_classes,text='Consultar classes',command=self.consultar_classes).pack(side='left')
        self.frame_prompts = ttk.Frame(topo)
        ttk.Label(self.frame_prompts,text='Classes em inglês').pack(side='left')
        ttk.Entry(self.frame_prompts,textvariable=self.prompts).pack(side='left',padx=(10,0),fill='x',expand=True)
        ttk.Label(topo,text='COCO: 0=person, 2=car, 5=bus, 7=truck. Empilhadeira e trator: prefira YOLOE.').pack(anchor='w')
        lim = ttk.Frame(topo)
        lim.pack(fill='x',pady=(13,4))
        for titulo, var, largura in [ ('Confiança mínima',self.conf_min,8),('Confiança alta',self.conf_alta,8),('Imagens por lote',self.lote,6),('Limite (0=todas)',self.limite,7) ]:
            ttk.Label(lim,text=titulo).pack(side='left',padx=(0,5))
            ttk.Entry(lim,textvariable=var,width=largura).pack(side='left',padx=(0,19))
        checks = ttk.Frame(topo)
        checks.pack(fill='x',pady=(6,9))
        ttk.Checkbutton(checks,text='Incluir subpastas',variable=self.recursivo).pack(side='left')
        ttk.Checkbutton(checks,text='SIMULAÇÃO: não mover arquivos',variable=self.simular).pack(side='left',padx=30)
        ttk.Label(topo,text='Detecção ≥ confiança alta | Dúvida: entre mínima e alta | Sem detecções: nenhuma classe-alvo ≥ mínima.').pack(anchor='w')
        botoes = ttk.Frame(topo)
        botoes.pack(fill='x',pady=(14,7))
        self.iniciar_botao = ttk.Button(botoes,text='▶ INICIAR TRIAGEM',command=self.iniciar)
        self.iniciar_botao.pack(side='left')
        self.parar_botao = ttk.Button(botoes,text='■ PARAR após imagem atual',command=self.parar,state='disabled')
        self.parar_botao.pack(side='left',padx=10)
        ttk.Label(botoes,textvariable=self.status).pack(side='right')
        ttk.Progressbar(topo,variable=self.progress,maximum=100).pack(fill='x',pady=5)
        ttk.Label(topo,text='Log da execução').pack(anchor='w',pady=(9,4))
        frame_log = ttk.Frame(topo)
        frame_log.pack(fill='both',expand=True)
        self.log = tk.Text(frame_log,wrap='word',height=12,bg='#121920',fg='#f3f7fc',insertbackground='white',font=('Consolas',10))
        barra = ttk.Scrollbar(frame_log,orient='vertical',command=self.log.yview)
        self.log.configure(yscrollcommand=barra.set)
        self.log.pack(side='left',fill='both',expand=True)
        barra.pack(side='right',fill='y')
        self.logar('Instruções: execute notebook v7 no Colab, copie a URL e X-API-Key para esta janela.')
        self.logar('Recomendação: mantenha SIMULAÇÃO marcada e limite 25 no primeiro teste.')

    def logar(self, mensagem):
        self.log.insert('end',mensagem+'\n')
        self.log.see('end')

    def alternar_modelo(self):
        self.frame_classes.pack_forget()
        self.frame_prompts.pack_forget()
        if self.modelo.get() == 'yolo26':
            self.frame_classes.pack(fill='x',pady=9,after=self.modelo_op)
        else:
            self.frame_prompts.pack(fill='x',pady=9,after=self.modelo_op)

    def escolher_pasta(self, var):
        pasta = filedialog.askdirectory(title='Selecione uma pasta')
        if pasta:
            var.set(pasta)
            if var is self.origem and not self.destino.get().strip():
                self.destino.set(str(Path(pasta).parent / (Path(pasta).name+'_triadas')))

    def config(self):
        try:
            minimo = float(self.conf_min.get().replace(',','.'))
            alto = float(self.conf_alta.get().replace(',','.'))
            lote = int(self.lote.get())
            limite = int(self.limite.get())
        except ValueError as exc:
            raise ValueError('Verifique limites de confiança, tamanho do lote e limite de imagens.') from exc
        c = Config(origem=Path(self.origem.get().strip()),destino=Path(self.destino.get().strip()),
                   url=self.url.get().strip(),chave=self.chave.get().strip(),modelo=self.modelo.get(),
                   classes=self.classes.get().strip(),prompts=self.prompts.get().strip(),
                   conf_min=minimo,conf_alta=alto,lote=lote,simular=self.simular.get(),
                   recursivo=self.recursivo.get(),limite=limite)
        c.validar()
        return c

    def iniciar(self):
        if self.executando:
            return
        try:
            config = self.config()
        except Exception as exc:
            messagebox.showerror('Configuração inválida',str(exc))
            return
        # Inicia diretamente, sem aviso de confirmação a cada execução real.
        # A simulação continua disponível como opção independente na interface.
        self.cancelamento.clear()
        self.progress.set(0)
        self.executando = True
        self.status.set('Simulando...' if config.simular else 'Processando e movendo imagens...')
        if not config.simular:
            self.logar('[EXECUÇÃO REAL] As imagens classificadas serão movidas para a pasta de saída.')
        self.iniciar_botao.configure(state='disabled')
        self.parar_botao.configure(state='normal')
        def worker():
            try:
                resultado = processar(config, registrar=lambda s: self.eventos.put(('log',s)),
                    progresso=lambda n,total: self.eventos.put(('progress',n,total)),cancelar=self.cancelamento)
                self.eventos.put(('finish',resultado))
            except Exception as exc:
                self.eventos.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def parar(self):
        self.cancelamento.set()
        self.status.set('Parando de forma segura...')

    def consultar_classes(self):
        if self.executando:
            return
        try:
            c = self.config()
        except ValueError as exc:
            messagebox.showerror('Configuração inválida',str(exc))
            return
        self.logar('Consultando classes COCO no Colab...')
        def consultar():
            try:
                api = APIColab(c)
                dados = api._requisicao('/classes',metodo='GET')
                self.eventos.put(('classes', dados.get('classes',[])))
            except Exception as exc:
                self.eventos.put(('log',f'[ERRO] Classes: {exc}'))
        threading.Thread(target=consultar,daemon=True).start()

    def mostrar_classes(self, classes):
        dialog = tk.Toplevel(self)
        dialog.title('Selecionar classes COCO')
        dialog.geometry('410x540')
        ttk.Label(dialog,text='Ctrl+clique para selecionar várias classes').pack(pady=12)
        lista = tk.Listbox(dialog,selectmode='extended',font=('Consolas',11))
        lista.pack(fill='both',expand=True,padx=15)
        ids = []
        for item in classes:
            ids.append(int(item['id']))
            lista.insert('end',f"{item['id']:>2}   {item['name']}")
        for idx, item in enumerate(ids):
            if str(item) in self.classes.get().split(','):
                lista.selection_set(idx)
        def aceitar():
            selecionadas = [str(ids[i]) for i in lista.curselection()]
            if selecionadas:
                self.classes.set(','.join(selecionadas))
                dialog.destroy()
            else:
                messagebox.showwarning('Seleção vazia','Selecione pelo menos uma classe.',parent=dialog)
        ttk.Button(dialog,text='Usar classes selecionadas',command=aceitar).pack(pady=12)

    def ler_eventos(self):
        try:
            while True:
                evento = self.eventos.get_nowait()
                if evento[0] == 'log':
                    self.logar(evento[1])
                elif evento[0] == 'progress':
                    _, n, total = evento
                    self.progress.set(n*100/total if total else 0)
                    self.status.set(f'{n}/{total} analisadas')
                elif evento[0] == 'classes':
                    self.mostrar_classes(evento[1])
                elif evento[0] in ('finish','error'):
                    self.executando = False
                    self.iniciar_botao.configure(state='normal')
                    self.parar_botao.configure(state='disabled')
                    if evento[0] == 'finish':
                        resp = evento[1]
                        self.status.set('Concluído')
                        self.logar('Relatório CSV: '+resp['csv'])
                        messagebox.showinfo('Triagem concluída',
                                            f"Processadas: {resp['processadas']} / {resp['total']}\n"
                                            f"Detecção: {resp['contagem']['deteccao']}\n"
                                            f"Dúvida: {resp['contagem']['duvida']}\n"
                                            f"Sem detecções: {resp['contagem']['sem_deteccoes']}\n"
                                            f"Erros: {resp['contagem']['erro']}\n\nCSV: {resp['csv']}")
                    else:
                        self.status.set('Interrompido por erro')
                        self.logar('[ERRO] '+evento[1])
                        messagebox.showerror('Falha na triagem',evento[1])
        except queue.Empty:
            pass
        self.after(120,self.ler_eventos)


if __name__ == '__main__':
    Aplicativo().mainloop()
