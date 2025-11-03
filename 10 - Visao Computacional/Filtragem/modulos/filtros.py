import cv2
import math
import numpy as np

# LER IMAGEM
def read_image(filename):
    # Lê uma imagem com a OpenCV
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # Verifica se é uma imagem colorida
    if len(image.shape) > 2: 
        # Convert de BGR para RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image  

# REDIMENSIONAR IMAGEM
def resize_image(image,width,height):
  # Verifica se é uma imagem colorida
    if len(image.shape) > 2: 
        rows, cols, channels = image.shape
    else:
        rows, cols = image.shape
    # Calcula a nova largura e arredonda para cima
    w = int(math.ceil(cols*width/100))
    # Calcula a nova altura e arredonda para  cima
    h = int(math.ceil(rows*height/100))
    # Define o novo tamanho como uma tupla
    new_size = (w,h)
    # Redimensiona com a OpenCV
    image = cv2.resize(image,new_size)
    return image

# CONVERTER PARA ESCALA DE CINZA
def grayscale_image(image):
  # Verifica se é uma imagem colorida
    if len(image.shape) > 2:
        # Converte com a OpenCV
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return image

# FILTRO DA MÉDIA
def average_filter(image,kernel_size):
    # Filtro da média da OpenCV recebendo uma imagem e uma tupla com tamanho do kernel
    image = cv2.blur(image,(kernel_size,kernel_size))
    return image

# FILTRO GAUSSIANO
def gaussian_filter(image,kernel_size):
    # O valor do desvio padrão é diretamente proporcional 
    # a intensidade da suavização da imagem
    # Desvio padrão = 0, a OpenCV utiliza um valor padrão para o desvio padrão
    # que é calculado automaticamente com base no tamanho do kernel.    
    standard_deviation = 0
    # Filtro Gaussiano da OpenCV recebendo a imagem, 
    # uma tupla com tamanho do kernel e o desvio padrão
    image = cv2.GaussianBlur(image,(kernel_size,kernel_size),standard_deviation)
    return image

# FILTRO DA MEDIANA
def median_filter(image,kernel_size):
    # Filtro da mediana da OpenCV recebendo uma imagem e o tamanho do kernel
    image = cv2.medianBlur(image,kernel_size)
    return image

# RUÍDO SAL E PIMENTA
def salt_and_pepper_noise(image):
    # Cria uma cópia da imagem original, pois as operações de aplicação do
    # ruído modificam os valores dos pixels dieretamente na imagem
    # o que afetaria a imagem original carregada pela interface gráfica
    image_noisy = image.copy()  
    # Verifica se é uma imagem colorida
    if len(image_noisy.shape) > 2: 
        # Obtém as dimensões e os canais da imagem colorida
        rows,cols,channels = image_noisy.shape
    else: 
        # Obtém as dimensõesda imagem escala de cinza
        rows,cols = image_noisy.shape
    # Cria uma matriz para armazenar o ruído com as mesmas dimensões de image_noisy
    noise = np.zeros((rows,cols),np.uint8)
    # Preenche a matriz de ruído com valores aleatórios entre 0 e 255
    cv2.randu(noise,0,255)
    # Aplica o ruído 'sal e pimenta' em image_noisy:
    # Define os pixels com valor <= 5 como pretos (0)
    image_noisy[noise <= 5] = 0
    # Define os pixels com valor >= 250 como brancos (255)
    image_noisy[noise >= 250] = 255
    return image_noisy

# FILTRO DE SOBEL
def sobel_filter(image):
  # Verifica se é uma imagem colorida
    if len(image.shape) > 2:
        # Converte para escala de cinza
        gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)  
    else:
        # Se já estiver em escala de cinza
        # gray_image recebe a imagem sem precisar converte-la
        gray_image = image
    # Detectar bordas na direção x (verticais):
    # Utiliza o filtro Sobel para calcular a derivada em relação a x.
    # a derivada retorna pontos onde há grandes mudanças de intensidade dos pixels (bordas)
    # cv2.Sobel() é uma função do OpenCV que aplica o filtro Sobel.
    # Parâmetros:
    # - gray_image: a imagem de entrada em escala de cinza.
    # - cv2.CV_64F: a profundidade do destino, ou seja, o tipo do dado do resultado. 
    #   CV_64F significa que o resultado terá pixels com valores de ponto flutuante de 64 bits.
    # - 1, 0: os parâmetros dx e dy, que definem a ordem da derivada x e y respectivamente. 
    #   Aqui, dx=1 e dy=0 significa que estamos calculando a derivada na direção x.
    # - ksize=3: o tamanho da janela do filtro Sobel (kernel size). Aqui, uma janela de 3x3 é usada.
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F,1,0,ksize=3)
    # Detectar bordas na direção y (horizontais):
    # Similar ao filtro Sobel aplicado na direção x, mas agora estamos calculando a derivada em relação a y.
    # Parâmetros:
    # - gray_image: a imagem de entrada em escala de cinza.
    # - cv2.CV_64F: a profundidade do destino, ou seja, o tipo do dado do resultado.
    # - 0, 1: os parâmetros dx e dy, que definem a ordem da derivada x e y respectivamente. 
    #   Aqui, dx=0 e dy=1 significa que estamos calculando a derivada na direção y.
    # - ksize=3: o tamanho da janela do filtro Sobel (kernel size). Aqui, uma janela de 3x3 é usada.
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F,0,1,ksize=3)
    # Calcular a magnitude dos gradientes:
    # Combina as derivadas nas direções x e y para obter a magnitude do gradiente em cada pixel.
    # a magnitude do gradiente retorna a intensidade do gradiente (ou borda) em cada pixel, independentemente da direção
    # - np.square(sobel_x): eleva ao quadrado cada valor do gradiente x.
    # - np.square(sobel_y): eleva ao quadrado cada valor do gradiente y.
    # - np.sqrt(): calcula a raiz quadrada da soma dos quadrados dos gradientes x e y, obtendo a magnitude do gradiente.
    # O resultado é uma imagem onde cada pixel representa a intensidade do gradiente (bordas) na posição correspondente.
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    # Normalizar para o intervalo [0,255]
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    # Converte o resultado do gradiente para valores inteiros de 8 bits sem sinal [0,255]
    # e armazena em uma imagem que contém apenas as bordas obtidas pelo filtro de Sobel
    edge_image = np.uint8(gradient_magnitude)
    # Verifica a imagem ORIGINAL é colorida
    if len(image.shape) > 2: 
        # Converte a imagem de BORDAS para o formato RGB
        edge_image = cv2.cvtColor(edge_image,cv2.COLOR_GRAY2RGB) 
    # Combina a imagem original com a imagem de bordas a através da operação aritmética de adição
    # O resultado é uma nova imagem onde as bordas são sobrepostas na imagem original
    image_add = cv2.add(image,edge_image)
    return image_add
    
# FILTRO LAPLACIANO
def laplacian_filter(image):
    # Aplica um filtro de mediana com uma janela de 3x3 à imagem para suavizar e reduzir o ruído
    image = median_filter(image,3)
    # Define o kernel do filtro Laplaciano, que é utilizado para detectar bordas e realçar detalhes na imagem
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    # Aplica o filtro Laplaciano à imagem usando a função filter2D do OpenCV
    # Esta função realiza a convolução da imagem com o kernel Laplaciano
    # O parâmetro -1 indica que a imagem de saída terá a mesma profundidade (número de canais) da imagem de entrada
    laplacian = cv2.filter2D(image,-1,kernel)
    # Adiciona a imagem original com o resultado do filtro Laplaciano
    # Isso realça as bordas detectadas e aumenta o contraste em áreas de alta frequência (regiões com mudanças abruptas)
    image_add = cv2.add(image,laplacian)
    return image_add

# FILTRO HIGH BOOST
def highboost_filter(image,a):
    # Aplica um filtro de mediana com uma janela de 3x3 à imagem para suavizar e reduzir o ruído
    image = median_filter(image,3)
    # Define o kernel do filtro High Boost, que é utilizado para detectar bordas e realçar detalhes na imagem
    # a é um parâmetro ajustável onde a  > 0 destaca ainda mais as bordas e a < 0 atenua as bordas
    kernel = np.array([[0, -1, 0],
                       [-1, 4+a, -1],
                       [0, -1, 0]])
    # Aplica o filtro High Boost à imagem usando a função filter2D do OpenCV
    # Esta função realiza a convolução da imagem com o kernel High Boost
    # O parâmetro -1 indica que a imagem de saída terá a mesma profundidade (número de canais) da imagem de entrada                       
    highboost = cv2.filter2D(image,-1,kernel)
    # Adiciona a imagem original com o resultado do filtro High Boost
    # Isso realça as bordas detectadas e aumenta o contraste em áreas de alta frequência (regiões com mudanças abruptas)
    image_add = cv2.add(image,highboost)
    return image_add
