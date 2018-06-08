# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 23:43:12 2018

Note that: 
    This code is translated from the MATLAB version of minresQLP:
        http://www.stanford.edu/group/SOL/software.html
        
Contact: 
    yang.liu15(AT)uqconnect.edu.au
    
Author:
    Yang Liu, 
    School of Mathematics and Physics, 
    The University of Queensland.
    
Advisor: 
    Farbod Roosta-Khorasani,
    School of Mathematics and Physics,
    The University of Queensland.

REFERENCES:
    S.-C. Choi, C. C. Paige, and M. A. Saunders,
    MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric
    systems, SIAM Journal of Scientific Computing, submitted on March 7, 2010.

    S.-C. Choi's PhD Dissertation, Stanford University, 2006:
        http://www.stanford.edu/group/SOL/dissertations.html

--------------------------------------------------------------------------
minresQLP: Aim to obtain the min-length solution of symmetric 
   (possibly singular) Ax=b or min||Ax-b||.

   X = minresQLP(A,B) solves the system of linear equations A*X=B
   or the least-squares problem min norm(B-A*X) if A is singular.
   The N-by-N matrix A must be symmetric or Hermitian, but need not be
   positive definite or nonsingular.  It may be double or single.
   The rhs vector B must have length N.  It may be real or complex,
   double or single, 

   X = minresQLP(AFUN,B) accepts a function handle AFUN instead of
   the matrix A.  Y = AFUN(X) returns the matrix-vector product Y=A*X.
   In all of the following syntaxes, A can be replaced by AFUN.

   X = minresQLP(A,B,RTOL) specifies a stopping tolerance.
   If RTOL=[] or is absent, a default value is used.
   (Similarly for all later input parameters.)
   Default RTOL=1e-6.

   X = minresQLP(A,B,RTOL,MAXIT)
   specifies the maximum number of iterations.  Default MAXIT=N.

   X = minresQLP(A,B,RTOL,MAXIT,M)
   uses a matrix M as preconditioner.  M must be positive definite
   and symmetric or Hermitian.  It may be a function handle MFUN
   such that Y=MFUN(X) returns Y=M\X.
   If M=[], a preconditioner is not applied.

   X = minresQLP(A,B,RTOL,MAXIT,M,SHIFT)
   solves (A - SHIFT*I)X = B, or the corresponding least-squares problem
   if (A - SHIFT*I) is singular, where SHIFT is a real or complex scalar.
   Default SHIFT=0.

   X = minresQLP(A,B,RTOL,MAXIT,M,SHIFT,MAXXNORM,ACONDLIM,TRANCOND)
   specifies three parameters associated with singular or
   ill-conditioned systems (A - SHIFT*I)*X = B.

   MAXXNORM is an upper bound on NORM(X).
   Default MAXXNORM=1e7.

   ACONDLIM is an upper bound on ACOND, an estimate of COND(A).
   Default ACONDLIM=1e15.

   TRANCOND is a real scalar >= 1.
   If TRANCOND>1,        a switch is made from MINRES iterations to
                         MINRES-QLP iterationsd when ACOND >= TRANCOND.
   If TRANCOND=1,        all iterations will be MINRES-QLP iterations.
   If TRANCOND=ACONDLIM, all iterations will be conventional MINRES
                         iterations (which are slightly cheaper).
   Default TRANCOND=1e7.

   X = minresQLP(A,B,RTOL,MAXIT,M,SHIFT,MAXXNORM,ACONDLIM,TRANCOND,SHOW)
   specifies the printing option.
   If SHOW=true,  an iteration log will be output.
   If SHOW=false, the log is suppressed.
   Default SHOW=true.
   

   FLAG:
   -1 (beta2=0)  B and X are eigenvectors of (A - SHIFT*I).
    0 (beta1=0)  B = 0.  The exact solution is X = 0.                    
    1 X solves the compatible (possibly singular) system (A - SHIFT*I)X = B
      to the desired tolerance:
         RELRES = RNORM / (ANORM*XNORM + NORM(B)) <= RTOL,
      where
              R = B - (A - SHIFT*I)X and RNORM = norm(R).
    2 X solves the incompatible (singular) system (A - SHIFT*I)X = B
      to the desired tolerance:
         RELARES = ARNORM / (ANORM * RNORM) <= RTOL,
      where
              AR = (A - SHIFT*I)R and ARNORM = NORM(AR).
    3 Same as 1 with RTOL = EPS.
    4 Same as 2 with RTOL = EPS.
    5 X converged to an eigenvector of (A - SHIFT*I).
    6 XNORM exceeded MAXXNORM.
    7 ACOND exceeded ACONDLIM.
    8 MAXIT iterations were performed before one of the previous
      conditions was satisfied.
    9 The system appears to be exactly singular.  XNORM does not
      yet exceed MAXXNORM, but would if further iterations were
      performed.

    ITER:  the number of iterations performed. ITER = MITER + QLPITER.
    MITER:  the number of conventional MINRES iterations.
    QLPITER:  the number of MINRES-QLP iterations.
    
    RELRES & RELARES: Relative residuals for (A - SHIFT*I)X = B and the 
            associated least-squares problem.  RELRES and RELARES are 
            defined above in the description of FLAG.
            
    ANORM:  an estimate of the 2-norm of A-SHIFT*I.
    ACOND:  an estimate of COND(A-SHIFT*I,2).
    XNORM:  a recurred estimate of NORM(X).
    AXNORM:   a recurred estimate of NORM((A-SHIFT*I)X)
    
    RESVEC:  a vector of estimates of NORM(R) at each iteration,
             including NORM(B) as the first entry.
    ARESVEC: a vector of estimates of NORM((A-SHIFT*I)R) at each
             iteration, including NORM((A-SHIFT*I)B) as the first entry.
    RESVEC and ARESVEC have length ITER+1.
 
COPYRIGHT NOTICE:
   If you seek permission to copy and distribute translations of this 
   software into another language, please e-mail a specific request to
   saunders@stanford.edu and scchoi@stanford.edu.
 
"""

import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv, norm
from scipy.sparse.linalg import cg
from scipy.sparse.linalg.interface import aslinearoperator

def MinresQLP(A, b, rtol, maxit, M=None, shift=None, maxxnorm=None,
              Acondlim=None, TranCond=None, show=False, rnormvec=False):
    
    A = aslinearoperator(A)
    if shift is None:
        shift = 0
    if maxxnorm is None:
        maxxnorm = 1e7
    if Acondlim is None:
        Acondlim = 1e15
    if TranCond is None:
        TranCond = 1e7
    if rnormvec:
        resvec = []
        Aresvec = []
        
    
    n = len(b)
    b = b.reshape(n,1)
    r2 = b
    r3 = r2
    beta1 = norm(r2)
    
    if M is None:
        noprecon = True
        pass
    else:
        noprecon = False
        r3 = Precond(M, r2)
        beta1 = r3.T.dot(r2) #teta
        if beta1 <0:
            print('Error: "M" is indefinite!')
        else:
            beta1 = np.sqrt(beta1)
    
    ## Initialize
    flag0 = -2
    flag = -2
    iters = 0
    QLPiter = 0
    beta = 0
    tau = 0
    taul = 0
    phi = beta1
    betan = beta1
    gmin = 0
    cs = -1
    sn = 0
    cr1 = -1
    sr1 = 0
    cr2 = -1
    sr2 = 0
    dltan = 0
    eplnn = 0
    gama = 0
    gamal = 0
    gamal2 = 0
    eta = 0
    etal = 0
    etal2 = 0
    vepln = 0
    veplnl = 0
    veplnl2 = 0
    ul3 = 0
    ul2 = 0
    ul = 0
    u = 0
    rnorm = betan
    xnorm = 0
    xl2norm = 0
    Axnorm = 0
    Anorm = 0
    Acond = 1
    relres = rnorm / (beta1 + 1e-50)
    x = np.zeros((n,1))
    w = np.zeros((n,1))
    wl = np.zeros((n,1))
    if rnormvec:
        resvec = np.append(resvec, beta1)
    
    msg = [' beta2 = 0.  b and x are eigenvectors                   ',  # -1
           ' beta1 = 0.  The exact solution is  x = 0               ',  # 0
           ' A solution to Ax = b found, given rtol                 ',  # 1
           ' Min-length solution for singular LS problem, given rtol',  # 2
           ' A solution to Ax = b found, given eps                  ',  # 3
           ' Min-length solution for singular LS problem, given eps ',  # 4
           ' x has converged to an eigenvector                      ',  # 5
           ' xnorm has exceeded maxxnorm                            ',  # 6
           ' Acond has exceeded Acondlim                            ',  # 7
           ' The iteration limit was reached                        ',  # 8
           ' Least-squares problem but no converged solution yet    ']  # 9
    
    if show:
        print(' ')
        print('Enter Minres-QLP: ')
        print('Min-length solution of symmetric(singular)', end=' ')
        print('(A-sI)x = b or min ||(A-sI)x - b||')
        #||Ax - b|| is ||(A-sI)x - b|| if shift != 0 here
        hstr1 = '    n = %8g    ||Ax - b|| = %8.2e     ' % (n, beta1) 
        hstr2 = 'shift = %8.2e       rtol = %8g' % (shift, rtol)
        hstr3 = 'maxit = %8g      maxxnorm = %8.2e  ' % (maxit, maxxnorm)
        hstr4 = 'Acondlim = %8.2e   TranCond = %8g' % (Acondlim, TranCond)
        print(hstr1, hstr2)
        print(hstr3, hstr4)
        
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 0
    
    while flag == flag0 and iters < maxit:
        #lanczos
        iters += 1
        betal = beta
        beta = betan
        v = r3/beta
        r3 = Ax(A, v)
        if shift == 0:
            pass
        else:
            r3 = r3 - shift*v
        
        if iters > 1:
            r3 = r3 - r1*beta/betal
        
        alfa = np.real(r3.T.dot(v))
        r3 = r3 - r2*alfa/beta
        r1 = r2
        r2 = r3
        
        if noprecon:
            betan = norm(r3)
            if iters == 1:
                if betan == 0:
                    if alfa == 0:
                        flag = 0
                        break
                    else:
                        flag = -1
                        x = b/alfa
                        break
        else:
            r3 = Precond(M, r2)
            betan = r2.T.dot(r3)
            if betan > 0:
                betan = np.sqrt(betan)
            else:
                print('Error: "M" is indefinite or singular!')
        pnorm = np.sqrt(betal ** 2 + alfa ** 2 + betan ** 2)
        
        #previous left rotation Q_{k-1}
        dbar = dltan
        dlta = cs*dbar + sn*alfa
        epln = eplnn
        gbar = sn*dbar - cs*alfa
        eplnn = sn*betan
        dltan = -cs*betan
        dlta_QLP = dlta
        #current left plane rotation Q_k
        gamal3 = gamal2
        gamal2 = gamal
        gamal = gama
        cs, sn, gama = SymGivens(gbar, betan)
        gama_tmp = gama
        taul2 = taul
        taul = tau
        tau = cs*phi
        Axnorm = np.sqrt(Axnorm ** 2 + tau ** 2)
        phi = sn*phi
        #previous right plane rotation P_{k-2,k}
        if iters > 2:
            veplnl2 = veplnl
            etal2 = etal
            etal = eta
            dlta_tmp = sr2*vepln - cr2*dlta
            veplnl = cr2*vepln + sr2*dlta
            dlta = dlta_tmp
            eta = sr2*gama
            gama = -cr2 *gama
        #current right plane rotation P{k-1,k}
        if iters > 1:
            cr1, sr1, gamal = SymGivens(gamal, dlta)
            vepln = sr1*gama
            gama = -cr1*gama
        
        #update xnorm
        xnorml = xnorm
        ul4 = ul3
        ul3 = ul2
        if iters > 2:
            ul2 = (taul2 - etal2*ul4 - veplnl2*ul3)/gamal2
        if iters > 1:
            ul = (taul - etal*ul3 - veplnl *ul2)/gamal
        xnorm_tmp = np.sqrt(xl2norm**2 + ul2**2 + ul**2)
        if abs(gama) > np.finfo(np.double).tiny and xnorm_tmp < maxxnorm:
            u = (tau - eta*ul2 - vepln*ul)/gama
            if np.sqrt(xnorm_tmp**2 + u**2) > maxxnorm:
                u = 0
                flag = 6
        else:
            u = 0
            flag = 9
        xl2norm = np.sqrt(xl2norm**2 + ul2**2)
        xnorm = np.sqrt(xl2norm**2 + ul**2 + u**2)
        #update w&x
        #Minres
        if (Acond < TranCond) and flag != flag0 and QLPiter == 0:
            wl2 = wl
            wl = w
            w = (v - epln*wl2 - dlta_QLP*wl)/gama_tmp
            if xnorm < maxxnorm:
                x += tau*w
            else:
                flag = 6
        #Minres-QLP
        else:
            QLPiter += 1
            if QLPiter == 1:
                xl2 = np.zeros((n,1))
                if (iters > 1):  # construct w_{k-3}, w_{k-2}, w_{k-1}
                    if iters > 3:
                        wl2 = gamal3*wl2 + veplnl2*wl + etal*w
                    if iters > 2:
                        wl = gamal_QLP*wl + vepln_QLP*w
                    w = gama_QLP*w
                    xl2 = x - wl*ul_QLP - w*u_QLP
                    
            if iters == 1:
                wl2 = wl
                wl = v*sr1
                w = -v*cr1                
            elif iters == 2:
                wl2 = wl
                wl = w*cr1 + v*sr1
                w = w*sr1 - v*cr1
            else:
                wl2 = wl
                wl = w
                w = wl2*sr2 - v*cr2
                wl2 = wl2*cr2 +v*sr2
                v = wl*cr1 + w*sr1
                w = wl*sr1 - w*cr1
                wl = v
            xl2 = xl2 + wl2*ul2
            x = xl2 + wl*ul + w*u         

        #next right plane rotation P{k-1,k+1}
        gamal_tmp = gamal
        cr2, sr2, gamal = SymGivens(gamal, eplnn)
        #transfering from Minres to Minres-QLP
        gamal_QLP = gamal_tmp
        #print('gamal_QLP=', gamal_QLP)
        vepln_QLP = vepln
        gama_QLP = gama
        ul_QLP = ul
        u_QLP = u
        ## Estimate various norms
        abs_gama = abs(gama)
        Anorml = Anorm
        Anorm = max([Anorm, pnorm, gamal, abs_gama])
        if iters == 1:
            gmin = gama
            gminl = gmin
        elif iters > 1:
            gminl2 = gminl
            gminl = gmin
            gmin = min([gminl2, gamal, abs_gama])
        Acondl = Acond
        Acond = Anorm / gmin
        rnorml = rnorm
        relresl = relres
        if flag != 9:
            rnorm = phi
        relres = rnorm / (Anorm * xnorm + beta1)
        rootl = np.sqrt(gbar ** 2 + dltan ** 2)
        Arnorml = rnorml * rootl
        relAresl = rootl / Anorm
        ## See if any of the stopping criteria are satisfied.
        epsx = Anorm * xnorm * np.finfo(float).eps
        if (flag == flag0) or (flag == 9):
            t1 = 1 + relres
            t2 = 1 + relAresl
            if iters >= maxit:
                flag = 8 #exit before maxit
            if Acond >= Acondlim:
                flag = 7 #Huge Acond
            if xnorm >= maxxnorm:
                flag = 6 #xnorm exceeded
            if epsx >= beta1:
                flag = 5 #x = eigenvector
            if t2 <= 1:
                flag = 4 #Accurate Least Square Solution
            if t1 <= 1:
                flag = 3 #Accurate Ax = b Solution
            if relAresl <= rtol:
                flag = 2 #Trustful Least Square Solution
            if relres <= rtol:
                flag = 1 #Trustful Ax = b Solution
        if flag == 2 or flag == 4 or flag == 6 or flag == 7:
            #possibly singular
            iters = iters - 1
            Acond = Acondl
            rnorm = rnorml
            relres = relresl
        else:
            if rnormvec:
                resvec = np.append(resvec, rnorm)
                Aresvec = np.append(Aresvec, Arnorml)
                
            if show:
                if iters%10 - 1 == 0:
                    lstr = ('        iter     rnorm    Arnorm    relres   ' +
                            'relAres    Anorm     Acond     xnorm')
                    print(' ')
                    print(lstr)
                if QLPiter == 1:
                    print('QLP', end='')
                else:
                    print('   ', end='')
                lstr1 = '%8g    %8.2e ' % (iters-1, rnorml)
                lstr2 = '%8.2e  %8.2e ' % (Arnorml, relresl)
                lstr3 = '%8.2e  %8.2e ' % (relAresl, Anorml)
                lstr4 = '%8.2e  %8.2e ' % (Acondl, xnorml)
                print(lstr1, lstr2, lstr3, lstr4)
            
    #exited the main loop
    if QLPiter == 1:
        print('QLP', end = '')
    else:
        print('   ', end = '')
    Miter = iters - QLPiter
    
    #final quantities
    r1 = b - Ax(A,x) + shift*x
    rnorm = norm(r1)
    Arnorm = norm(Ax(A,r1) - shift*r1)
    xnorm = norm(x)
    relres = rnorm/(Anorm*xnorm + beta1)
    relAres = 0
    if rnorm > np.finfo(np.double).tiny:
        relAres = Arnorm/(Anorm*rnorm)    
        
    if show:
        if rnorm > np.finfo(np.double).tiny:                
            lstr1 = '%8g    %8.2e ' % (iters, rnorm)
            lstr2 = '%8.2eD %8.2e ' % (Arnorm, relres)
            lstr3 = '%8.2eD %8.2e ' % (relAres, Anorm)
            lstr4 = '%8.2e  %8.2e ' % (Acond, xnorm)
            print(lstr1, lstr2, lstr3, lstr4)
        else:                
            lstr1 = '%8g    %8.2e ' % (iters, rnorm)
            lstr2 = '%8.2eD %8.2e ' % (Arnorm, relres)
            lstr3 = '          %8.2e ' % (Anorm)
            lstr4 = '%8.2e  %8.2e ' % (Acond, xnorm)
            print(lstr1, lstr2, lstr3, lstr4)
        
        print(' ')
        print('Exit Minres-QLP: ')
        str1 = 'Flag = %8g    %8s' % (flag, msg[int(flag + 1)])
        str2 = 'Iter = %8g      ' % (iters)
        str3 = 'Minres = %8g       Minres-QLP = %8g' % (Miter, QLPiter)
        str4 = 'relres = %8.2e    relAres = %8.2e    ' % (relres, relAres)
        str5 = 'rnorm = %8.2e      Arnorm = %8.2e' % (rnorm, Arnorm)
        str6 = 'Anorm = %8.2e       Acond = %8.2e    ' % (Anorm, Acond)
        str7 = 'xnorm = %8.2e      Axnorm = %8.2e' % (xnorm, Axnorm)
        print(str1)
        print(str2, str3)
        print(str4, str5)
        print(str6, str7)
        
    if rnormvec:
        Aresvec = np.append(Aresvec, Arnorm)
        return (x,flag,iters,Miter,QLPiter,relres,relAres,Anorm,Acond,
                xnorm,Axnorm,resvec,Aresvec)
    
    return (x,flag,iters,Miter,QLPiter,relres,relAres,Anorm,Acond,xnorm,Axnorm)


def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax = A.dot(x)
    return Ax

def Precond(M, r):
    if callable(M):
        h = cg(M, r)
    else:
        h = inv(M).dot(r)
    return h

def SymGivens(a, b):    
    if b == 0:
        if a == 0:
            c = 1
        else:
            c = np.sign(a)
        s = 0
        r = abs(a)
    elif a == 0:
        c = 0
        s = np.sign(b)
        r = abs(b)
    elif abs(b) > abs(a):
        t = a / b
        s = np.sign(b) / np.sqrt(1 + t ** 2)
        c = s * t
        r = b / s
    else:
        t = b / a
        c = np.sign(a) / np.sqrt(1 + t ** 2)
        s = c * t
        r = a / c
    return c, s, r

def main(): 
##################    example1    ####################
    n=100
    e = np.ones((n,1))
    data = np.c_[-2*e,4*e,-2*e]
    A = sp.spdiags(data.T, [-1,0,1],n,n).toarray()
    M = sp.spdiags(4*e.T, 0,n,n).toarray()
    b = sum(A)
    rtol = 1e-10
    maxit = 50
    x = MinresQLP(A,b,rtol,maxit,M,show=True)
#    x = MinresQLP(A,b,rtol,maxit,M,show=True,rnormvec=True)
#    print(x[11])
#    print(x[12])
    
##################    example2    ####################
#    n=50
#    N=n**2
#    e = np.ones((n,1))
#    data = np.c_[e, e, e]
#    B = sp.spdiags(data.T, [-1,0,1],n,n)
#    A_mid = np.array([]).reshape(0,0)
#    for i in range(n):
#        A_mid = sp.block_diag((A_mid, B))
#        if i == 0:
#            A_upper = sp.hstack([sp.csr_matrix((n,n)), B])
#            A_lower = sp.vstack([sp.csr_matrix((n,n)), B])            
#        if i > 0 and i < n-1:
#            A_upper = sp.block_diag((A_upper, B))
#            A_lower = sp.block_diag((A_lower, B))
#        if i == n-1:
#            A_upper = sp.vstack([A_upper, sp.csr_matrix((n,N))])
#            A_lower = sp.hstack([A_lower, sp.csr_matrix((N,n))])  
#    A = A_upper + A_lower + A_mid
#    b = sum(A.toarray())
#    rtol = 1e-5
#    x = MinresQLP(A, b, rtol, N, maxxnorm = 1e2, show = True)        
        
##################    example3    ####################
#    a = -10
#    c = -a
#    n = 2*c + 1
#    A = sp.spdiags(np.arange(a, c+1), 0, n, n)
#    b = np.ones((n, 1))
#    rtol = 1e-6
#    x = MinresQLP(A, b, rtol, n, maxxnorm = 1e2, show = True)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    