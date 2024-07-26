import math

######################################################################################
# Descriptive statistics
######################################################################################

def mean(X):
    return sum(X)/len(X)

def variance(X):
    mu = mean(X)
    return sum([(x - mu)**2 for x in X])/len(X)

def sample_variance(X):
    mu = mean(X)
    return sum([(x - mu)**2 for x in X])/(len(X)-1)

def raw_moment(X,k):
    return sum([x**k for x in X])/len(X)

def central_moment(X,k):
    mu = mean(X)
    return sum([(x - mu)**k for x in X])/len(X)

def percentile(X, k):
    N = len(X)
    n = math.ceil(k/100*(N-1))
    return sorted(X)[n]

def median(X):
    return percentile(X, 50)

def mode(X):
    count_dict = {}
    for x in X:
        if x in count_dict:
            count_dict[x] += 1
        else:
            count_dict[x] = 1
    max_cont = -1
    mode = None
    
    for key, value in count_dict.items():
        if value > max_cont:
            mode = key
            max_cont = value
    
    return mode

def coefficient_of_variation(X):
    return variance(X)**0.5/mean(X)

def descriptive_statistics_summary(X):
    return {
        'Mean' : mean(X),
        'Variance' : variance(X),
        'Standard Deviation': variance(X)**0.5,
        'Coefficient of Variation': coefficient_of_variation(X),
        'Skewness' : central_moment(X,3),
        'Kurtosis' : central_moment(X,4),
        '0.25 Percentile' : percentile(X, 25),
        'Median' : median(X),
        '0.75 Percentile' : percentile(X,75),
        'Mode' : mode(X)
    }

def covariance(X,Y):
    XY = []
    for i in range(len(X)):
        XY.append(X[i]*Y[i])

    return mean(XY)-mean(X)*mean(Y)

def pearson_coeff(X,Y):
    return covariance(X,Y)/( variance(X)**0.5*variance(Y)**0.5 )

# Multivariate analysis

def mean_vector(X):
    m = len(X[0])
    mu_vect = []
    for i in range(m):
        mu_vect.append(mean([row[i] for row in X]))
    return mu_vect

def covariance_matrix(X):
    m = len(X[0])
    Cov = [[0 for i in range(m)] for j in range(m)]
    for i in range(m):
        for j in range(i,m):
            x_1 = [row[i] for row in X]
            x_2 = [row[j] for row in X]
            c = covariance(x_1,x_2)
            Cov[i][j] = c
            Cov[j][i] = c
    return Cov

def correlation_matrix(X):
    m = len(X[0])
    Corr = [[0 for i in range(m)] for j in range(m)]
    for i in range(m):
        for j in range(i,m):
            x_1 = [row[i] for row in X]
            x_2 = [row[j] for row in X]
            c = pearson_coeff(x_1,x_2)
            Corr[i][j] = c
            Corr[j][i] = c
    return Corr

data_matrix = [
    [4.0, 2.0, 0.6],
    [4.2, 2.1, 0.59],
    [3.9, 2.0, 0.58],
    [4.3, 2.1, 0.62],
    [4.1, 2.2, 0.63]
]

##################################################################################################
# Common Probability distributions
##################################################################################################

# Probability density functions
def normal_pdf(x):
    return (2*math.pi)**(-0.5)*math.exp(-0.5*(x**2))

def t_student_pdf(nu, x):
    return math.gamma(0.5 * (nu + 1.0)) / math.gamma(0.5 * nu) / (nu * math.pi)**0.5 * (1 + x**2/nu)**(-0.5*(nu + 1))

def chi_squared_pdf(n, x):
    return 1/(2**(n/2)*math.gamma(n/2))*x**(n/2-1)*math.exp(-x/2)

def F_snedecor_pdf(n1,n2,x):
    return (((n1*x)**n1*n2**n2)/(n1*x+n2)**(n1+n2))**0.5/(x*math.gamma(n1/2)*math.gamma(n2/2)/math.gamma((n1+n2)/2))
def beta_pdf(a,b,x):
    return x**(a-1)*(1-x)**(b-1)/(math.gamma(a)*math.gamma(b)/math.gamma(a+b))
    
# Probability mass functions
def binomial_pmf(n, p, x):
    return math.comb(n,x)*p**x*(1-p)**(n-x)

def poisson_pmf(l, x):
    return math.exp(-l)*l**x/math.factorial(x)

#######################################################################################################
# Abscissas
#######################################################################################################

# Returns the abscissa z_{alpha} such that P(Z > z_{alpha}) = alpha
def normal_abscissa(alpha):
    # Aproximates the integral of the pdf, and stops when the condition is satisfied
    a = 0
    b = 0.0001
    dx = (b-a)
    probability = 1
    integral = 0
    
    while probability > alpha:
        f_a = normal_pdf(a)
        f_b = normal_pdf(b)
        integral += f_b*dx + (f_a - f_b)*dx/2
        probability = 1 - (0.5 + integral)
        a = b
        b += dx
    
    return b

# Returns the abscissa t_{nu,alpha} such that P(T > t_{nu,alpha}) = alpha
def t_student_abscissa(nu, alpha):
    a = 0
    b = 0.0001
    dx = (b-a)
    probability = 1
    integral = 0
    while probability > alpha:
        f_a = t_student_pdf(nu,a)
        f_b = t_student_pdf(nu,b)
        integral += f_b*dx + (f_a - f_b)*dx/2
        probability = 1 - (0.5 + integral)
        a = b
        b += dx
        
    return b

# Returns the abscissa x_{n,alpha} such that P(X > x_{n,alpha}) = alpha
def chi_squared_abscissa(n, alpha):
    a = 0
    b = 0.0001
    dx = (b-a)
    probability = 1
    integral = 0
    while probability > alpha:
        f_a = chi_squared_pdf(n,a)
        f_b = chi_squared_pdf(n,b)
        integral += f_b*dx + (f_a - f_b)*dx/2
        probability = 1 -  integral
        a = b
        b += dx
    
    return b

# Returns the abscissa x_{n1,n2,alpha} such that P(X > x_{n1,n2,alpha} = alpha)
def beta_abscissa(n1,n2,alpha):
    a = 0
    b = 0.00005
    dx = (b-a)
    probability = 1
    integral = 0
    while probability > alpha:
        f_a = beta_pdf(n1,n2,a)
        f_b = beta_pdf(n1,n2,b)
        integral += f_b*dx + (f_a - f_b)*dx/2
        probability = 1 -  integral
        a = b
        b += dx
    
    return b

def f_snedecor_abscissa(n1,n2, alpha):
    x = beta_abscissa(n1/2,n2/2,alpha)
    return n2*x/(n1*(1-x))



###################################################################################################
# Confidence intervals
###################################################################################################

# Returns the interval [a,b] such that the probability of the mean of the data set X be in [a,b] is (1 - alpha)
def normal_mean_confidence_interval(X,  alpha, standard_deviation = None):
    # Computes the mean of the data set X
    mu = mean(X)

    # The user supose that the data follows a Normal distribution
    if standard_deviation is None:
        # Computes the sample standard deviation
        S = sample_variance(X)
        # Interval
        t = t_student_abscissa(len(X)-1,alpha/2)
        a = mu - t*S/n**(0.5)
        b = mu + t*S/n**(0.5)
    # The user doesn't supose that the data follows a Normal distribution (this works better for large samples with n > 30)
    else:
        # Interval endpoints
        z = normal_abscissa(alpha/2)
        a = mu - z*standard_deviation/n**(0.5)
        b = mu + z*standard_deviation/n**(0.5)

    return [a, b]

def normal_mean_difference_confidence_interval(X, Y, alpha):
    x = mean(X)
    y = mean(Y)
    s_x = variance(X)
    s_y = variance(Y)
    n_x = len(X)
    n_y = len(Y)

    S_p = ( ((n_x - 1)*s_x + (n_y - 1)*s_y)/(n_x + n_y - 2) )**0.5
    if n_x >= 30 and n_y >= 30:
        z = normal_abscissa(alpha/2)
        a = (x - y) - z*S_p*(1/n_x + 1/n_y)**0.5
        b = (x - y) + z*S_p*(1/n_x + 1/n_y)**0.5
    else:
        t = t_student_abscissa(n_x + n_y - 2, alpha/2)
        a = (x - y) - t*S_p*(1/n_x + 1/n_y)**0.5
        b = (x - y) + t*S_p*(1/n_x + 1/n_y)**0.5
    
    return [a,b]

def binomial_proportion_confidence_interval(X, alpha):
    # X is a vector of 1's and 0's where 1 means the element satisfy the property studied (Bernoulli)
    n = len(X)
    p = sum(X)/n
    
    z = normal_abscissa(alpha/2)
    a = (p - z/n**0.5*(p*(1-p))**0.5)
    b = (p + z/n**0.5*(p*(1-p))**0.5)

    return [a, b]

def poisson_mean_confidence_interval(X, alpha):
    mu = mean(X)

    # Interval endpoints
    z = normal_abscissa(alpha/2)
    a = mu - z*(mu/n)**0.5
    b = mu - z*(mu/n)**0.5

    return [a,b]

def normal_variance_confidence_interval(X, alpha, mean=None):
    n = len(X)
    if mean is None:
        mu = mean(X)
        S = sample_variance(X)
    
        a = (n-1)*S/(chi_squared_abscissa(n-1, alpha/2))
        b = (n-1)*S/(chi_squared_abscissa(n-1, 1-  alpha/2))

        return [a, b]
    k = 0
    for i in range(n):
        k += (X[i] - mu)**2
    
    a = k / chi_squared_abscissa(n,alpha/2)
    b = k / chi_squared_abscissa(n, 1 - alpha/2)

    return [a, b]

