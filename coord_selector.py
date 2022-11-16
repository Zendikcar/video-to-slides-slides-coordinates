import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


debug = True

def findSlide(image):
    '''
    Individua in modo autonomo la slide nell'immagine (esclude le bande nere ai lati dell'immagine).
    Restituisce la posizione della slide e le sue dimensioni in proporzione alle dimensioni 
    dell'immagine
    '''
    imgray    = np.sum(image, axis=2) / 3
    cmean     = np.sum(imgray, axis=0) / np.shape(imgray)[0]
    size      = np.shape(cmean)[0]
    step      = 5
    max_delta = 25
    old_col   = np.array([0, 0])
    new_col   = np.array([step, size-step])
    delta     = np.array([0.0, 0.0])

    for i in range(0,2) :
        while np.abs(delta[i]) < max_delta and np.abs(new_col[i] - size/2) > 2*step :
            old_col[i] = new_col[i]
            new_col[i]+= (1-2*i)*step
            delta[i]   = cmean[new_col[i]] - cmean[old_col[i]]

    pos = np.array([new_col[0]/size, 0.0])
    dim = np.array([(new_col[1]-new_col[0])/size, 1.0])
    return (pos, dim)


def onselect(eclick, erelease):
	''' 
	Funzione chiamata quando si preme o si rilascia il mause sopra l'immagine.
	Se il debug è abilitato stampa le coordinate del punto in alto a sinistra
	e in basso a destra del rettangolo di selezione.
	'''
	print("P1:", int(eclick.xdata), int(eclick.ydata))
	print("P2:", int(erelease.xdata), int(erelease.ydata))


def select_coord(image):
	'''
	Data un immagine permette permette la selezione manuale della numerazione della slide, 
	interagendo con la finestra di matplotlib.
	Restituisce i vertici in alto a sinistra e in basso a destra del rettangolo di selezione
	in proporzione alle dimensioni dell'immagine di partenza
	'''
	fig, ax = plt.subplots(figsize=(16, 9))
	ax.imshow(image)
	ax.axis('off')
	props = dict(facecolor='blue', alpha=0.5)
	rect  = RectangleSelector(ax, onselect, 
		useblit=True,
		button=[1, 3],
		minspanx=5, minspany=5,
		spancoords='pixels',
		interactive=True, 
		props=props)
	plt.show()

	xcoords,ycoords = rect.corners
	wid, hei = np.shape(image)[1], np.shape(image)[0]
	p1 = [xcoords[0] / wid, ycoords[0] / hei]
	p2 = [xcoords[2] / wid, ycoords[2] / hei]

	return (p1, p2)


def run_test_selection():
	'''
	Esempio di utilizzo in combo delle funzioni di tracciamento delle slide (automatico) 
	e della numerazione delle slide (manuale)
	'''
	image    = cv2.imread("./slide_test.jpeg")
	if image == None:
		print("Image not found")
		return
	width    = np.shape(image)[1]
	pos, dim = findSlide(image)
	p1,  p2  = select_coord(image[:, int(pos[0]*width):int((pos[0]+dim[0])*width), :])
	print(p1, p2)