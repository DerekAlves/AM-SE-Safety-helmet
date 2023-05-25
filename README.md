# AM-SE-Safety-helmet
Projeto para a disciplina de aprendizagem de máquina em sistemas embarcados envolvendo classificação de imagens de pessoas portando ou não o capacete de segurança.

O modelo de rede neural convolucional foi treinado no google colab e pode ser encontrado no arquivo .ipynb.

Para embarcar o modelo utilizamos micropython juntamente com o firmware microlite para espcam32 (firmware.bin), o código embarcado em micropython está em HelmetInferencing.py e o arquivo image_arr.py contém quatro imagens descritas em arrays do python, o arquivo safe.tflite é o modelo convertido para tensorflow lite.

Utilizamos o Thonny para embarcar, para iniciar as inferências se faz necessário importar o arquivo HelmetInferencing.py para iniciar as inferências, desta maneira:

```py
>>> import HelmetInferencing
```
