import math

########################################################################################################################################

# Common Probability distributions

#######################################################################################################################################

# Probability density functions
def normal_pdf(x):
    return (2*math.pi)**(-0.5)*math.exp(-0.5*(x**2))

def t_student_pdf(nu, x):
    return math.gamma(0.5 * (nu + 1.0)) / math.gamma(0.5 * nu) / math.sqrt(nu * math.pi) * (1 + x**2/nu)**(-0.5*(nu + 1))

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

###########################################################################################################################################

# Abscissas

###########################################################################################################################################
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
        b += 0.0001
    
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
        b += 0.0001
        
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
        b += 0.0001
    
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
        b += 0.00005
    
    return b

def f_snedecor_abscissa(n1,n2, alpha):
    x = beta_abscissa(n1/2,n2/2,alpha)
    return n2*x/(n1*(1-x))



#####################################################################################################################################

# Confidence intervals

####################################################################################################################################

# Returns the interval [a,b] such that the probability of the mean of the data set X be in [a,b] is (1 - alpha)
def normal_mean_confidence_interval(X,  alpha, standard_deviation = None):
    # Computes the mean of the data set X
    mean = 0
    n = len(X)
    for i in range(n):
        mean += X[i]/n
   
    # The user supose that the data follows a Normal distribution
    if standard_deviation is None:
        # Computes the sample standard deviation
        V = 0
        for i in range(len(X)):
            V += (X[i] - mean)**2
        S = (V / (n-1))**(0.5)
        # Interval
        a = mean - t_student_abscissa(len(X)-1,alpha/2)*S/n**(0.5)
        b = mean + t_student_abscissa(len(X)-1,alpha/2)*S/n**(0.5)
    # The user doesn't supose that the data follows a Normal distribution (this works better for large samples with n > 30)
    else:
        # Interval endpoints
        a = round(mean - normal_abscissa(alpha/2)*standard_deviation/n**(0.5), 4)
        b = round(mean + normal_abscissa(alpha/2)*standard_deviation/n**(0.5), 4)

    return [a, b]

def normal_mean_difference_confidence_interval(X, Y, alpha):
    x = mean(X)
    y = mean(Y)
    s_x = variance(X)
    s_y = variance(Y)
    n_x = len(X)
    n_y = len(Y)

    S_p = sqrt( ((n_x - 1)*s_x + (n_y - 1)*s_y)/(n_x + n_y - 2) )
    if n_x >= 30 and n_y >= 30:
        a = (x - y) - normal_abscissa(alpha/2)*S_p*sqrt(1/n_x + 1/n_y)
        b = (x - y) + normal_abscissa(alpha/2)*S_p*sqrt(1/n_x + 1/n_y)
    else:
        a = (x - y) - t_student_abscissa(n_x + n_y - 2, alpha/2)*S_p*sqrt(1/n_x + 1/n_y)
        b = (x - y) + t_student_abscissa(n_x + n_y - 2, alpha/2)*S_p*sqrt(1/n_x + 1/n_y)
    
    return [a,b]

def binomial_proportion_confidence_interval(X, alpha):
    # X is a vector of 1's and 0's where 1 means the element satisfy the property studied (Bernoulli)
    n = len(X)
    p = sum(X)/n

    a = (p - normal_abscissa(alpha/2)/sqrt(n)*sqrt(p*(1-p)))
    b = (p + normal_abscissa(alpha/2)/sqrt(n)*sqrt(p*(1-p)))

    return [a, b]

def poisson_mean_confidence_interval(X, alpha):
    # Computes the mean of the data set X
    mean = 0
    n = len(X)
    for i in range(n):
        mean += X[i]/n

    # Interval endpoints
    a = round(mean - normal_abscissa(alpha/2)*(mean/n)**0.5, 4)
    b = round(mean - normal_abscissa(alpha/2)*(mean/n)**0.5, 4)

    return [a,b]

def normal_variance_confidence_interval(X, alpha, mean=None):
    n = len(X)
    if mean is None:
        mean = 0
        for i in range(n):
            mean += X[i]/n
        variance_sample = 0
        for i in range(n):
            variance_sample += (X[i] - mean)**2/(n-1)
        a = (n-1)*variance_sample/(chi_squared_abscissa(n-1, alpha/2))
        b = (n-1)*variance_sample/(chi_squared_abscissa(n-1, 1-  alpha/2))

        return [a, b]
    k = 0
    for i in range(n):
        k += (X[i] - mean)**2
    
    a = round(k / chi_squared_abscissa(n,alpha/2), 4)
    b = round(k / chi_squared_abscissa(n, 1 - alpha/2), 4)

    return [a, b]

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

def median(X):
    X_sorted = sorted(X)
    n = len(X)
    if n % 2 == 1:
        return X_sorted[n // 2]
    else:
        return (X_sorted[n // 2 - 1] + X_sorted[n // 2 + 1])/2

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
        'Median' : median(X),
        'Mode' : mode(X)
    }

def covariance(X,Y):
    XY = []
    for i in range(len(X)):
        XY.append(X[i]*Y[i])

    return mean(XY)-mean(X)*mean(Y)
