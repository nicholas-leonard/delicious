import numpy
import pylab

def dispims(M, height, width, border=0, bordercolor=0.0, layout=None, **kwargs):
    """ Display a stack of vectorized matrices (colunmwise). 

    """
    numimages = M.shape[1]
    if layout is None:
        n0 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
        n1 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * numpy.ones(((height+border)*n0+border,(width+border)*n1+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[i*(height+border)+border:(i+1)*(height+border)+border,
                   j*(width+border)+border :(j+1)*(width+border)+border] = numpy.vstack((
                            numpy.hstack((numpy.reshape(M[:,i*n1+j],(height, width)),
                                   bordercolor*numpy.ones((height,border),dtype=float))),
                            bordercolor*numpy.ones((border,width+border),dtype=float)
                            ))
    pylab.imshow(im, cmap=pylab.cm.gray, interpolation='nearest', **kwargs)
    pylab.show()

