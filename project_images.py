import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from PIL import Image

'''
Loads the dataset from a provided .npy file, re-centres it around
the origin, and returna it as a numpy array of floats.
'''
def load_and_centre_dataset(filename):
    data = np.load(filename).astype('float32')
    data -= np.mean(data, axis=0) # re-centering
    return data
    return data

'''
Returns covariance matrix
'''
def get_covariance(dataset):
    return np.cov(np.transpose(dataset))

'''
To find m largest eigenvectors and eigenvalues from S
Use subset_by_value or subset_by_index
Expected values in writeup
'''
def get_eig(S, m):
    n = len(S)
    lam, U = eigh(S, eigvals_only=False, subset_by_index=[n-m, n-1])
    lam = lam[::-1]*np.identity(m) # reverse to decreasing order
    for i in range(n):
        U[i] = U[i][::-1] # reverse to decreasing order
    return lam, U

'''
Returns the set of eigenvectors/values that explains more than 
'prop' proportion of the covariance
'''
def get_eig_prop(S, prop):
    n = len(S)
    lam, U = eigh(S, eigvals_only=False)
    lam = lam[::-1] # reverse to decreasing order
    for i in range(n):
        U[i] = U[i][::-1] # reverse to decreasing order

    # calculating proportion of variance based on lambda values
    trace = np.sum(lam)
    props = []
    for i in range(n):
        if lam[i]/trace > prop:
            props.insert(i, lam[i]/trace)
            
    m = len(props) # m largest eigenvalues have needed proportion
    return lam[0:m]*np.identity(m), U[:,0:m]

'''
Returns projection of original image using eigenvectors
'''
def project_image(image, U):
    image_pro = 0
    (m, n) = np.shape(U)
    for i in range(n):
        image_pro += np.dot(np.dot(np.transpose(U[:,i]), image), U[:,i])
    return image_pro

'''
Displays original and projected images
'''
def display_image(orig, proj):
    # Reshaping/formatting image vectors
    orig = np.reshape(orig, (32, 32))  
    proj = np.reshape(proj, (32, 32))
    orig = np.rot90(orig, k=3) 
    proj = np.rot90(proj, k=3)

    # Creating a figure with two subplots
    fig, axes = plt.subplots(1,2)
    # Setting titles
    axes[0].set_title('Original')
    axes[1].set_title('Projection')

    ax1 = axes[0].imshow(orig, aspect='equal')
    ax2 = axes[1].imshow(proj, aspect='equal') 
    # Setting colorbar size to match plots
    fig.colorbar(ax1, ax=axes[0], fraction=0.045)
    fig.colorbar(ax2, ax=axes[1], fraction=0.045)
    fig.tight_layout(pad=3.0) # Increase padding between plots
    plt.show()

def main():
    filename = 'YaleB_32x32.npy'
    data = load_and_centre_dataset(filename)
    S = get_covariance(data)
    prop = 0.05 # can be increased to improve clarity
    lam, U = get_eig_prop(S, prop)
    original = data[0] # can be changed to pick a different original picture from data
    projection = project_image(original, U)
    display_image(original, projection)

main()