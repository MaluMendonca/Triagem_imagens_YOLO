"""Execucao por terminal; permite processar milhares de imagens sem janela GUI."""
import argparse
from pathlib import Path
from triagem_core import Config, processar

parser = argparse.ArgumentParser(description='Triagem automatica YOLO com API Colab')
parser.add_argument('--origem',required=True,type=Path)
parser.add_argument('--destino',required=True,type=Path)
parser.add_argument('--url',required=True)
parser.add_argument('--chave',required=True)
parser.add_argument('--modelo',default='yolo26',choices=['yolo26','yoloe26'])
parser.add_argument('--classes',default='0')
parser.add_argument('--prompts',default='')
parser.add_argument('--conf-min',type=float,default=0.12)
parser.add_argument('--conf-alta',type=float,default=0.45)
parser.add_argument('--lote',type=int,default=4)
parser.add_argument('--limite',type=int,default=25)
parser.add_argument('--sem-recursao',action='store_true')
parser.add_argument('--mover',action='store_true',help='Move os originais. Sem isso, apenas simula!')
args = parser.parse_args()

config = Config(origem=args.origem,destino=args.destino,url=args.url,chave=args.chave,
                modelo=args.modelo,classes=args.classes,prompts=args.prompts,
                conf_min=args.conf_min,conf_alta=args.conf_alta,lote=args.lote,
                limite=args.limite,simular=not args.mover,recursivo=not args.sem_recursao)
resultado=processar(config)
print('RESULTADO:',resultado)
