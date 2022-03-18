#!/usr/bin/env python
# coding: utf-8

# # Assignment A1 [35 marks]
# 
# The assignment consists of 3 questions. Each exercise may contain coding and/or discussion questions.
# - Type your **code** in the **code cells** provided below each question.
# - For **non-code** questions, use the **Markdown cells** provided below each question, indicated by 📝. Double-click these cells to edit them, and run them to display your Markdown-formatted text. Please refer to the Week 1 tutorial notebook for Markdown syntax.

# ---
# ## Question 1: Estimating $\pi$ [8 marks]
# 
# Consider the 3 following formulas:
# 
# $$
# \begin{align}
# (1) \qquad &\prod_{n=1}^\infty \frac{4n^2}{4n^2-1} = \frac{\pi}{2} \\
# (2) \qquad &\sum_{n=0}^\infty \frac{2^n n!^2}{(2n+1)!} = \frac{\pi}{2} \\
# (3) \qquad &\sum_{n=1}^\infty \frac{(-1)^{n+1}}{n(n+1)(2n+1)} = \pi - 3
# \end{align}
# $$
# 
# Each of these formulas can be used to compute the value of $\pi$ to arbitrary precision, by computing as many terms of the partial sum or product as are required.
# 
# **1.1** Compute the sum or product of the first $m$ terms for each formula, with $m = 1, 2, 3, \dots, 30$.
# 
# Present your results graphically, using 2 plots, both with the total number of terms on the x-axis.
# 
# - The first plot should display the value of the partial sum or product for different numbers of terms, and clearly indicate the exact value of $\pi$ for reference.
# - The second plot should display the absolute value of the error between the partial sum or product and $\pi$, with the y-axis set to logarithmic scale.
# 
# **[5 marks]**

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(15, 5)) # Create a subplot

list0 = [] # The list for the value of x-axis(n)
for i in range(1,31):
    list0.append(i)

list1 = [2*(4*(1**2))/(4*(1**2)-1)] # The list for the value of First Formula 
t = 4*(1**2)/(4*(1**2)-1)
for i in range(2,31):
    t *= (4*(i**2))/(4*(i**2)-1)
    list1.append(2*t)

list2 = [] # The list for the value of pi  
for i in range(1,31):
    list2.append(np.pi)
    
list3 = [2*((2**0)*(math.factorial(0)**2))/(math.factorial(2*0+1))] # The list for the value of Second Formula 
h = ((2**0)*(math.factorial(0)**2))/(math.factorial(2*0+1))
for i in range(1,30):
    h += ((2**i)*(math.factorial(i)**2))/(math.factorial(2*i+1))
    list3.append(2*h)
    
list4 = [((-1)**(1+1))/((1)*(1+1)*(2*1+1))+3] # The list for the value of Third Formula 
S = ((-1)**(1+1))/((1)*(1+1)*(2*1+1))
for i in range(2,31):
    S += ((-1)**(i+1))/((i)*(i+1)*(2*i+1))
    list4.append(S+3)

ax.set_xlim([0, 31]) 
ax.set_ylim([1.9,3.3]) 
ax.set_xlabel('Number of terms', fontsize=14)
ax.set_ylabel('Partial sums/products', fontsize=14)
plt.plot(list0, list2,'r-', label = r'Exact value of Pi', marker = 'o') 
plt.plot(list0, list1,'g--', label = r'Partial products of Pi(First Formula)',marker = 'o')
plt.plot(list0, list3,'b--', label = r'Partial sums of Pi(Second Formula)',marker = 'o')
plt.plot(list0, list4,'y--', label = r'Partial sums of Pi(Third Formula)',marker = 'o')
ax.legend(loc = 'lower right', fontsize = 14)
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(15, 5)) # Create a subplot

list0 = [] # The list for the value of x-axis(n)
for i in range(1,31):
    list0.append(i)

list5 = [np.log(abs((2*(4*(1**2))/(4*(1**2)-1))-np.pi))] # The list for the log(abs(error)) value between pi and First Formula 
t = 4*(1**2)/(4*(1**2)-1)
for i in range(2,31):
    t *= (4*(i**2))/(4*(i**2)-1)
    list5.append(np.log(abs((2*t)-np.pi)))

list6 = [np.log(abs((2*((2**0)*(math.factorial(0)**2))/(math.factorial(2*0+1)))-np.pi))] # The list for the log(abs(error)) value between pi and Second Formula 
h = ((2**0)*(math.factorial(0)**2))/(math.factorial(2*0+1))
for i in range(1,30):
    h += ((2**i)*(math.factorial(i)**2))/(math.factorial(2*i+1))
    list6.append(np.log(abs((2*h)-np.pi)))

list7 = [np.log(abs((((-1)**(1+1))/((1)*(1+1)*(2*1+1))+3)-np.pi))] # The list for the log(abs(error)) value between pi and Third Formula 
S = ((-1)**(1+1))/((1)*(1+1)*(2*1+1))
for i in range(2,31):
    S += ((-1)**(i+1))/((i)*(i+1)*(2*i+1))
    list7.append(np.log(abs(S+3-np.pi)))

ax.set_xlim([0, 31])
ax.set_ylim([-25,1])
ax.set_xlabel('Number of terms', fontsize=14)
ax.set_ylabel('log(abs(error))', fontsize=14)
plt.plot(list0, list5,'g--', label = r'Error between Pi and First Formula',marker = 'o')
plt.plot(list0, list6,'b--', label = r'Error between Pi and Second Formula',marker = 'o')
plt.plot(list0, list7,'y--', label = r'Error between Pi and Third Formula',marker = 'o')
ax.legend(loc = 'lower left', fontsize = 14)
plt.show()


# **1.2** If you did not have access to e.g. `np.pi` or `math.pi`, which of these 3 formulas would you choose to efficiently calculate an approximation of $\pi$ accurate to within any given precision (down to machine accuracy -- i.e. not exceeding $\sim 10^{-16}$)?
# 
# Explain your reasoning in your own words, with reference to your plots from **1.1**, in no more than 200 words.
# 
# **[3 marks]**

# 📝 _Use this cell to answer **1.2**_
# 
# I think the **Second formula** is more accurate to approxiamte $\pi$.
# Since we know that the value of log decreases with the value inside the log, and from the second plot, we can see that the log value of the Second formula is much smaller than other two formulas when n=30. And also after n=30, it still decreses constantly and it means it approaches $\pi$ more and more closer, but for other two formulas, there are about to convergent when n increases.

# ---
# ## Question 2: Numerical Linear Algebra [12 marks]
# 
# A **block-diagonal matrix** is a square matrix with square blocks of non-zero values along the diagonal, and zeros elsewhere. For example, the following matrix A is an example of a block-diagonal matrix of size $7\times 7$, with 3 diagonal blocks of size $2\times 2$, $3\times 3$, and $2 \times 2$ respectively:
# 
# $$
# A =
# \begin{pmatrix}
# 1 & 3 & 0 & 0 & 0 & 0 & 0 \\
# 2 & 2 & 0 & 0 & 0 & 0 & 0 \\
# 0 & 0 & -1 & 1 & 2 & 0 & 0 \\
# 0 & 0 & 2 & 1 & 0 & 0 & 0 \\
# 0 & 0 & 4 & 3 & 3 & 0 & 0 \\
# 0 & 0 & 0 & 0 & 0 & 4 & -2 \\
# 0 & 0 & 0 & 0 & 0 & 5 & 3
# \end{pmatrix}.
# $$
# 
# 
# **2.1** The code below creates a block-diagonal matrix with random non-zero values between 0 and 1, where all blocks have identical size. Study the following documentation sections:
# 
# - [`scipy.linalg.block_diag()` - SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.block_diag.html)
# - [`numpy.split()` - NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.split.html)
# - [Unpacking Argument Lists - Python tutorial](https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists)
# 
# Then, write detailed code comments in the cell below to explain **each line** of code, in your own words.
# 
# **[3 marks]**

# In[3]:


import numpy as np # Import the module 'numpy' as 'np'
from scipy.linalg import block_diag # Import the module 'scipy.linalg' as 'block_diag'

def random_blocks(m, shape): # Define the function 'random_blocks(m, shape)'
    '''
    Returns a list of m random matrices of size shape[0] x shape[1].
    '''
    
    mat = np.random.random([m * shape[0], shape[1]]) # Create a matrix which its size is equal to (m * shape[0], shape[1]),
                                                     # and the value of every element in this matrix is randomly in the half-open interval [0.0, 1.0). 
     
    blocks = np.split(mat, m) # Split the matrix 'mat' into m multiple arrays（blocks）
    
    
    return blocks # Return(output) the values 'blocks'


blocks = random_blocks(4, (3, 2)) # Randomly create a matrix which its size is equal to (12*2),
                                  # and the value of every element in this matrix is randomly in the half-open interval [0.0, 1.0),
                                  # and 'blocks' returns 4 splitted arrays(blocks) which are from the matrix 
                                  # and the size of every array(block) should be (3*2), 
                                  # and also they are in the order of the matrix.


A = block_diag(*blocks) # Create a block diagonal matrix 'A', 
                        # and the 4 splitted arrays(blocks) are arranging on the diagonal 
                        # and the remaining elements are all zero.


print(np.round(A, 3)) # Print 'A' and round every element in 3 decimal places.


# **2.2** For the rest of Question 2, we consider only block-diagonal matrices with $m$ blocks, where all diagonal blocks have the same shape $n \times n$. A block-diagonal system $Ax = b$ can be written as
# 
# $$
# \begin{pmatrix}
# A_{1} & & & \\
# & A_{2} & & \\
# & & \ddots & \\
# & & & A_{m}
# \end{pmatrix}
# \begin{pmatrix}
# x_1 \\ x_2 \\ \vdots \\ x_m
# \end{pmatrix}
# =
# \begin{pmatrix}
# b_1 \\ b_2 \\ \vdots \\ b_m
# \end{pmatrix}
# \qquad \Leftrightarrow \qquad
# \begin{cases}
# A_{1} x_1 &= b_1 \\
# A_{2} x_2 &= b_2 \\
# &\vdots \\
# A_{m} x_m &= b_m
# \end{cases},
# $$
# 
# where $A_i$ is the $i$th diagonal block of $A$, and $x_i$, $b_i$ are blocks of length $n$ of the vectors $x$ and $b$ respectively, for $i=1, 2, \dots, m$. Note that when $m=1$, this is a diagonal system.
# 
# We assume that all diagonal blocks $A_i$ are invertible, and therefore that the matrix $A$ is also invertible.
# 
# Write a function `linsolve_block_diag(blocks, b)` which takes 2 input arguments:
# 
# - `blocks`, a list of length $m$ storing a collection of $n \times n$ NumPy arrays (e.g. as returned by `random_blocks()` from **2.1**) representing the blocks $A_i$,
# - a NumPy vector `b` of length $mn$.
# 
# Your function should solve the block-diagonal system $Ax = b$, by solving **each sub-system $A_i x_i = b_i$ separately**, and return the solution as a NumPy vector `x` of length $mn$. You should choose an appropriate method seen in the course (e.g. `np.linalg.solve()`) to solve each sub-system.
# 
# **[3 marks]**

# In[4]:


import numpy as np

def linsolve_block_diag(blocks, b):
    '''
    Solves the block-diagonal system Ax=b,
    where the diagonal blocks are listed in "blocks".
    '''
    
    list_x = [] # Create an empty list for storing the value x(solution)
    m = len(blocks) 
    B = np.split(b,m) # Split b into m multiple arrays, so B has m multiple arrays
    
    for i in range(m): # Slove A_i * x_i = b_i m times and store every x_i into the list_x
        list_x.append(np.linalg.solve(blocks[i],B[i])) 
        
    return np.concatenate(list_x) # Make the list_x to a vector 

# Example testing for the function
blocks = [np.array([[1,2],[3,4]]),
          np.array([[5,6],[7,8]]),
          np.array([[9,10],[11,12]])]
b = np.array([11,21,31,41,51,61])
print(linsolve_block_diag(blocks, b))


# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# **2.3** We now wish to compare the computation time needed to solve a block-diagonal system $Ax = b$ using 2 different methods:
# 
# - solving the sub-systems one at a time, as in **2.2**,
# - solving the full system with a general method, not attempting to take the block-diagonal structure into account.
# 
# Consider block-diagonal systems with block sizes $n = 5, 10, 15, 20$, and a total number $m = 5, 10, 15, \dots, 40$ of blocks. For each combination of $n$ and $m$:
# 
# - Use the function `random_blocks()` from **2.1** to generate a list of $m$ random matrices of size $n\times n$.
# - Use the function `np.random.random()` to generate a random vector `b` of length $mn$.
# - Use your function `linsolve_block_diag()` from **2.2** to solve the system $Ax = b$, where $A$ is a block-diagonal matrix of size $mn \times mn$, with diagonal blocks given by the output of `random_blocks()`. Measure the computation time needed to solve the system.
# - Use the function `block_diag()` from `scipy.linalg` to form a NumPy array `A` of size $mn \times mn$, representing the block-diagonal matrix $A$.
# - Solve the full system $Ax = b$, using the same method you used in **2.2** for each individual sub-system. Measure the computation time needed to solve the system.
# 
# Create 4 plots, one for each value of $n$, to compare the computation time needed to solve $Ax=b$ with both methods, and how this varies with the total size of the system.
# 
# Summarise and discuss your observations in no more than 200 words.
# 
# **[6 marks]**

# In[8]:


import numpy as np
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

list_m = range(5,41,5) # Create a list_m which has m = 5,10,15,...,40
list_t = [] # Create an empty list_t to store the computation time
fig,ax = fig, ax = plt.subplots(1,4,figsize=(15, 8)) # Create 4 subplots 

for n in [5,10,15,20]: # Calculate n = 5 first, and then calculate n = 10,....
    for m in range(5,41,5): # Calculate m = 5 first, and then calculate m = 10,....
    
        t0 = time.time() # Create a time t0
        blocks = random_blocks(m, (n, n))
        b = np.random.random(m*n)
        linsolve_block_diag(blocks, b)
        t = time.time() - t0 # t is the computation time(use the present time - t0)
        
        list_t.append(t) # Store every t in the list_t
    
    c = int(n/5-1)
    ax[c].set_title(f'For n = {n}',fontsize = 15) # Set every subplot 
    ax[c].set_xlabel('m',fontsize = 15)
    ax[c].set_ylabel('Time (s)',fontsize = 15)
    ax[c].plot(list_m,list_t)
    list_t.clear() # When finish n = 5, clear the list_t, and then store t when n = 10,....

plt.subplots_adjust(hspace = 1, wspace = 0.8)
plt.show()


# 📝 _Use this cell to discuss your **2.3** results_
# 
# Firstly, for every plot, it shows an approximately linear relationship between the values of m and the computation time (When m increases, it spends more time on computing). Also for every plot, the maximum value on the y-axis increases with the values of n, it means that the computation time also increases with the values of n.

# ---
# ## Question 3: Numerical Integration [15 marks]
# 
# The integral of the function $f(x,y)= \sin(x) \cos\left(\frac{y}{5}\right)$ defined on the rectangle $\mathcal{D}\in\mathbb{R}^2 = (a,b)\times(c,d)$
# can be expressed in closed form as
# 
# $$
# I = \int_c^{d}\int_a^{b}  \sin(x)\cos\left(\frac{y}{5}\right) \ dx \ dy = 5\left(-\cos(b) + \cos(a)\right)\left(\sin\left(\frac{d}{5}\right) - \sin\left(\frac{c}{5}\right)\right).
# $$
# 
# for any $a<b$ and $c<d$.
# 
# **3.1** Create a surface plot of the function $f(x,y)$ on the interval $(-5, 5) \times (-5, 5)$.
# 
# **[3 marks]**

# In[9]:


from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(subplot_kw = {"projection":"3d"})

x = np.linspace(-5,5,500)
y = np.linspace(-5,5,500)
X,Y = np.meshgrid(x,y)
Z = np.sin(X)*np.cos(Y/5)

ax.plot_surface(X,Y,Z,cmap = cm.coolwarm)
plt.show()


# **3.2** Write a function `midpoint_I(D, N, M)` which takes 3 input arguments:
# 
# - a list `D` of length 4, to store the 4 interval bounds $a, b, c, d$,
# - a positive integer `N`,
# - a positive integer `M`,
# 
# and implements the 2D composite midpoint rule to compute and return an approximation of $I$, partitioning the rectangle $\mathcal{D}$ into $N\times M$ rectangles. (This translates to `N` nodes on the x-axis and `M` nodes on the y-axis.
# 
# You will need to adapt the 1D composite midpoint rule seen in Weeks 5 and 6 to extend it to 2 dimensions. Instead of approximating the integral by computing the sum of the surface areas of $N$ rectangles, you will need to sum the volumes of $N \times M$ cuboids.
# 
# **[3 marks]**

# In[10]:


import numpy as np

def midpoint_I(D, N, M):
    
    h_x = (D[1] - D[0])/N # The weight of x
    h_y = (D[3] - D[2])/M # The weight of y
    
    Volume = 0 
    
    for i in range(N): 
        for j in range(M):
            
            x_node = D[0] + h_x/2 + i*h_x # Every x_node is the midpoint (D[0] + h_x/2 is the first midpoint and then add one x_node is the second midpoint....)
            y_node = D[2] + h_y/2 + j*h_y # Every y_node is the midpoint (D[0] + h_y/2 is the first midpoint and then add one y_node is the second midpoint....)
        
            Volume += h_x*h_y*np.sin(x_node)*np.cos(y_node/5) # Add all cubes together to get the volume 
    
    return Volume

# Example testing for the function
print(f'The volume by midpoint rule on the interval (1,2)×(3,4) is {midpoint_I([1,2,3,4],10,20)}.\n')
print(f'The exact volume on the interval (1,2)×(3,4) is {5*(-np.cos(2) + np.cos(1))*(np.sin(4/5) - np.sin(3/5))}.')


# **3.3** Consider now the domain $\mathcal{D} = (0, 5)\times(0, 5)$. Compute the absolute error between the exact integral $I$ and the approximated integral computed with your `midpoint_I()` function from **3.2**, with all combinations of $M = 5, 10, 15, \dots, 300$ and $N = 5, 10, 15, \dots, 300$.
# 
# Store the error values in a $60\times 60$ NumPy array.
# 
# **[3 marks]**

# In[13]:


import numpy as np


Exact_Volume = 5*(-np.cos(5) + np.cos(0))*(np.sin(5/5) - np.sin(0/5))

def midpoint_I(D, N, M): # Same function as the above question
    
    h_x = (D[1] - D[0])/N
    h_y = (D[3] - D[2])/M
    
    Volume = 0
    
    for i in range(N):
        for j in range(M):
            
            x_node = D[0] + h_x/2 + i*h_x
            y_node = D[2] + h_y/2 + j*h_y
        
            Volume += h_x*h_y*np.sin(x_node)*np.cos(y_node/5)
    
    return Volume

NumPyarray_error = np.zeros([60, 60])

for M in range(5, 301, 5):
    for N in range(5, 301, 5): # Use Two loops to get all combinations of M and N  
        C = int(N/5-1)
        R = int(M/5-1)
        NumPyarray_error[R,C] = (abs(midpoint_I([0,5,0,5],M,N) - Exact_Volume)) # Store every error in the NumPyarray_error

print(f'The error values are {NumPyarray_error}')

# Require a couple of minutes to output


# **3.4** Display the absolute error values as a function of $N$ and $M$ using a contour plot, with contour levels at $10^{-k}, k = 1, 2, \dots, 5$.
# 
# You may find the documentation for [`contour()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html#matplotlib.pyplot.contour), [`contourf()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html#matplotlib.pyplot.contourf), and [colormap normalization](https://matplotlib.org/stable/tutorials/colors/colormapnorms.html#logarithmic) useful to clearly display your results.
# 
# **[3 marks]**

# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

list_MN = [] # Create a list_MN which has the values of x-axis and y-axis(M = 5,10,15,....; N = 5,10,15,....)
for i in range(5,301,5):
    list_MN.append(i) 

fig,ax = plt.subplots(figsize=(15, 5))

ax.contour(list_MN, list_MN, NumPyarray_error,levels = [10**(-5),10**(-4),10**(-3),10**(-2),10**(-1)],
           colors=['r', 'b', 'y','g','grey']) 
# Plot the contour diagram and blue curve is 10**(-4), yellow is 10**(-3), green is 10**(-2), and grey is 10**(-1).

ax.set_xlabel('N',fontsize=14) 
ax.set_ylabel('M',fontsize=14)
ax.set_xlim(0,310)
ax.set_ylim(-10,310)

plt.show()


# **3.5** Summarise and explain your observations from **3.4** in no more than 250 words. In particular, discuss how $\mathcal{D}$ should be partitioned to approximate $I$ with an error of at most $10^{-4}$, and comment on the symmetry or asymmetry of the partitioning.
# 
# **[3 marks]**

# 📝 _Use this cell to answer **3.5**_
# 
# From the plot, when N and M are increasing, the error is decreasing. Also, we see that four lines are becoming to stright horizontal lines after N = 50 approximately, so it means that the errors do not change a lot after N = 50 with fixed M. 
#  If M and N are up to 500, there may have the error of $10^{-5}$. The plot is asymmetric because the surface is not symmetric in $\mathcal{D}$.
