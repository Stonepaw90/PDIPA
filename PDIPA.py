#!/usr/bin/env python
# coding: utf-8

# # This is a program that runs the primal dual interior point algorithm given a function and a starting point  (and some other variables)
# ### by Abraham Holleran

#Anaconda
#cd /D C:\Users\Abraham\miniconda3\envs\snowflakes\Scripts
#streamlit run PDIPA.py

import numpy as np
import pandas as pd
import sympy
import math
import streamlit as st
from functools import reduce

st.set_page_config(layout = "wide")
# Edit these meta values if necessary.



alpha = 0.8 #step size parameter for getting away from constraint
beta = 0.9 #step size it parameter
epsilon = 0.001 #erative parameter
gamma = 0.1 #duality gapstopping tolerance


# Carefully put your variables, functions, and constraints here.

st.sidebar.button("Re Run")
alpha = st.sidebar.number_input("Alpha", 0.8, step = 0.1)
beta = st.sidebar.number_input("Beta", 0.9, step = 0.1)
epsilon = st.sidebar.number_input("Epsilon", 0.001, step = 0.001, format = "%f")
gamma = st.sidebar.number_input("Gamma", 0.1, step = 0.1)

st.title("Primal-dual Interior Point Algorithm")
st.header("By Abraham Holleran")
st.write("Written from the book [Linear and Convex Optimization](https://www.wiley.com/go/veatch/convexandlinearoptimization) under the supervision of the author, Dr. Michael Veatch.")
option = st.selectbox('Which problem do you want to optimize?', ('Example 9 (1 variable)', 'Example 10 (2 variables)'))
if option.split(' ')[1] == "10":
    option = 1
else:
    option = 2
if option == 1:
    st.latex(r'''\text{max } 10 + 10x_1 - 8x_2 - 4e^{x_1}-e^{x_1-x_2}''')
    st.latex(r'''\text{s.t.  } x_2 - x_1^{0.5} \leq 0''')
    st.latex(r'''-x_2 + x_1^{1.5} \leq 0 ''')
    x1,x2, mu = sympy.symbols('x1 x2 mu', real = True)
    X = sympy.Matrix([x1, x2])
    y1, y2 = sympy.symbols('y1 y2', real = True)
    Y = sympy.Matrix([y1, y2])
    all_vars = sympy.Matrix([x1, x2, y1, y2])
    f, g1, g2 = sympy.symbols('f g1 g2', cls=sympy.Function)
    s1, s2 = sympy.symbols('s1 s2', real = True)     #one s_i for each g_i, b_i
    #point = [0.164, 0.066, 5, 10]
    f = 10+10*x1 - 8*x2 - 4*sympy.exp(x1)-sympy.exp(x1-x2)
    g1 = x2-x1**(0.5)
    g2 = -x2 + x1**(1.5)
    g = sympy.Matrix([g1, g2])
    b = sympy.Matrix([0,0])
    alist = ["k", "mu", "x1", "x2", "y1", "y2", "f(x)", "lambda", "||d||"]
    st.write("Please write your (feasible) initial point.")
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    x1_input = col1.text_input("x1", "0.5")
    x2_input = col2.text_input("x2", "0.6")
    y1_input = col3.text_input("y1", "5.0")
    y2_input = col4.text_input("y2", "10.0")
    mu_input = col5.text_input("initial mu", "1.0")

    point = [float(x1_input), float(x2_input), float(y1_input), float(y2_input)]
    mu_value = float(mu_input)
elif option == 2:
    st.latex(r'''\text{max  } 10x-e^x''')
    st.latex(r'''\text{s.t.   } x \leq 2   ''')
    x1, x2, mu = sympy.symbols('x1 x2 mu', real=True)
    X = sympy.Matrix([x1])
    y1 = sympy.symbols('y1', real=True)
    Y = sympy.Matrix([y1])
    all_vars = sympy.Matrix([x1, y1])
    f, g1 = sympy.symbols('f g1 ', cls=sympy.Function)
    s1 = sympy.symbols('s1', real=True)  # one s_i for each g_i, b_i
    f = 10 * x1 - sympy.exp(x1)
    g1 = x1
    g = sympy.Matrix([g1])
    b = sympy.Matrix([2])
    alist = ["k", "mu", "x", "y", "f(x)", "lambda", "||d||"]
    st.write("Please write your (feasible) initial point.")
    col1, col2, col3 = st.beta_columns(3)
    point = [1,1]
    s = [1]
    while not all([i >= 0 for i in s]):

        error_found = True
    x1_input = col1.text_input("x", "1.0")
    y1_input = col2.text_input("y", "0.5")
    mu_input = col3.text_input("initial mu", "2.0")
    point = [float(x1_input), float(y1_input)]
    mu_value = float(mu_input)
s = sympy.Matrix([b[i] - g[i].subs([*zip(X, point[:len(X)])]).evalf() for i in range(len(g))])
input_point = point.copy()
assert all([i >= 0 for i in s]), f"The initial point does not satisfy the constraints. They have negative slacks [s1, s2] of {[*s]}."
assert all([i >= 0 for i in point[len(Y):]]), f"Choose positive y."

gradient = lambda f, v: sympy.Matrix([f]).jacobian(v)

H = sympy.hessian(f,X)
Z = sympy.zeros(len(X))
thing = [y_i*sympy.hessian(g_i, X)for y_i, g_i in zip(Y,g)]
sum_M = reduce((lambda x, y: x + y), thing)
Q = -H + sum_M


m = sympy.Matrix([mu/y_i for y_i in Y])
RHSB = b - g-m
J = g.jacobian(X)
RHST = gradient(f,X).T-J.T*Y
RHS = sympy.Matrix([RHST,RHSB])


st.latex(r'''\text{We need to solve for } \textbf{d}^x, \textbf{d}^y \text{ using (15.14): } \begin{bmatrix}
\textbf{Q} & \textbf{J}(\textbf{x})^T\\
\textbf{J}(\textbf{x}) & -\textbf{S}
\end{bmatrix}\begin{bmatrix}
\textbf{d}^x \\ \textbf{d}^y
\end{bmatrix}
=
\begin{bmatrix}
\nabla f(\textbf{x}) - \textbf{J}(\textbf{x})^T\textbf{y} \\
\textbf{b} - \textbf{g}(\textbf{x})-\textbf{m}
\end{bmatrix} ''')

#The following cell prints \\(\begin{bmatrix}
#\textbf{d}^x \\ \textbf{d}^y
#\end{bmatrix}\\).)
S = sympy.diag(*[(b_i - g_i)/y_i for b_i, g_i, y_i in zip(b, g, Y)])
LHS = sympy.Matrix([[Q, J.T], [J, -S]])



solv = LHS.LUsolve(RHS)
k = 0
done = False
data = []
shortcut = False
if option == 2:
    if st.button("Use lamda_max = (2-x)/d^x shortcut?"):
        shortcut = True
while not done and k < 14:
    #st.write("1")
    solv_eval = solv.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    f_eval = f.subs([*zip(X, point[:len(X)])])
    l_max1 =min(1,min([y_i/-dy_i if dy_i < 0 else 1 for y_i, dy_i in zip(point[-len(Y):], solv_eval[-len(Y):])]))
    if option ==2 and shortcut:
        l_max = min(l_max1, (2-point[0])/solv_eval[0])
    else:
        l_max = l_max1
    assert(all([y_i + l_max*dy_i >= 0 for y_i, dy_i in zip(point[-len(Y):], solv_eval[-len(Y):])]))
    #st.write("2")
    l= l_max
    all_constraints_satisfied = False
    iter = 0
    while not all_constraints_satisfied:
        violation = False
        iter += 1
        if iter > 10:
            exit(0)
        test_x = [i+l*j for i, j in zip(point[:len(X)], solv_eval[:len(X)])]
        for g_i, b_i in zip(g, b):
            g_eval = g_i.subs([*zip(all_vars[:len(X)], test_x)])
            if g_eval > b_i:
                violation = True

        #st.write(float(test_x[0]), float(g_eval), b)
        if violation:
            l *= beta
        else:
            all_constraints_satisfied = True
    l *= alpha
    #st.write("3")
    dnorm = math.sqrt(sum(map(lambda i : l*i * l*i, solv_eval[:len(X)])))
    mu_scientific = "{:2.1E}".format(mu_value)
    value_list = [k, mu_scientific, *[round(float(i),4) for i in point], round(f_eval,3), round(l,5), dnorm]
    data.append(value_list)
    point = [i+l*j for i,j in zip(point, solv_eval)]
    mu_value *= gamma
    k +=1
    #st.write("4")
    if math.sqrt(sum(map(lambda i : l*i * l*i, solv_eval[:len(X)]))) <= epsilon:
        st.write("""
        We're close enough as $\mid \mid \lambda$ **d**$^x \mid \mid \leq \epsilon$, indeed, """, round(dnorm, 6), """$\leq$""", epsilon, ".")
        done = True
df = pd.DataFrame(data, columns=alist)
st.write(df)
    #if k >= 5:
    #    st.write(l_max, type(l_max), alpha, beta, gamma, epsilon, mu_value)
rounded_point = [round(i, 4) for i in point]
st.write(f"The approximately optimal point is: {rounded_point}")
st.write(f"It has a value of: {round(f.subs([*zip(X, point[:len(X)])]),4)}.")

if st.button("Show symbolic matrices."):
    col6, col7 = st.beta_columns(2)
    col_help2 = 0
    matrix_list = [H, 1, Q, gradient(f, X), J.T * Y]
    matrix_string = ["\\nabla^2 f(\\textbf{x}) ", "0", "Q", "\\nabla f(\\textbf{x})", "J(\\textbf{x})^T\\textbf{y}"]
    for i in range(len(matrix_list)):
        if col_help2 % 2 == 0:
            with col6:
                if i == 1:
                    for j in range(len(g)):
                        st.latex("\\nabla^2 g_" + str(j + 1) + " (\\textbf{x}) =" + sympy.latex(g[j]))
                else:
                    st.latex(matrix_string[i] + " = " +sympy.latex(matrix_list[i]))
        else:
            with col7:
                if i == 1:
                    for j in range(len(g)):
                        st.latex("\\nabla^2 g_" + str(j + 1) + " (\\textbf{x}) =" + sympy.latex(g[j]))
                else:
                    st.latex(matrix_string[i] + " = " + sympy.latex(matrix_list[i]))
                    #latex_matrix1(matrix_string[i], matrix_list[i], False)
        col_help2 += 1
    with col7:
        st.latex("\\textbf{b}-\\textbf{g}-\\textbf{m} =" + sympy.latex(RHSB))
    if option == 2:
        st.latex(sympy.latex(LHS) + sympy.latex(sympy.Matrix(["d^x", "d_y"])) + "= " + sympy.latex(RHS))
        st.latex(sympy.latex(sympy.Matrix(["d^x", "d_y"])) + " = " + sympy.latex(solv))
    else:
        st.latex(
            sympy.latex(LHS) + sympy.latex(sympy.Matrix(["d_1^x", "d_2^x", "d_1^y", "d_2^y"])) + "= " + sympy.latex(RHS))





col4, col5 = st.beta_columns(2)

col_help = 0
def latex_matrix(name, matrix_for_me, col_bool, col_use1, col_use2):
    global col_help
    latex_string = name + " = " + "\\begin{bmatrix}  "
    shape_tuple = matrix_for_me.shape
    for i in range(len(matrix_for_me)):
        latex_string += str(matrix_for_me[i]) + " & "
        if ((i + 1) % shape_tuple[1] == 0):
            latex_string = latex_string[:-3] + " \\\\ "
    latex_string = latex_string[:-3] + "  \\end{bmatrix}"
    try:
        if col_bool:
            if col_help % 2 == 0:
                with col_use1:
                    st.latex(latex_string)
            else:
                with col_use2:
                    st.latex(latex_string)
        else:
            st.latex(latex_string)
    except:
        st.write("done broke")
    col_help += 1
    # print(latex_string)
def latex_matrix_sum(name, m1, m2, m3):
    latex_string = name + " = " + "\\begin{bmatrix}  "
    for i in range(len(m1)):
        latex_string += str(m1[i]) + " - " + str(m2[i]) + " - " + str(m3[i]) + " \\\\ "
    latex_string += " \\end{bmatrix} = \\begin{bmatrix}"
    new_thing = m1 - m2 - m3
    for i in new_thing:
        latex_string += str(i) + " \\\\ "
    latex_string = latex_string[:-2] + "  \\end{bmatrix}"
    st.latex(latex_string)
if st.button("Details of one numerical step."):
    col6, col7 = st.beta_columns(2)

    col_help = 0
    mu_value = mu_input
    point = input_point
    st.latex("\\text{We solve (15.14) numerically at the point } (\\textbf{x}_0, \\textbf{y}_0).")
    matrix_list = [H, 1, Q, gradient(f, X), J.T * Y]
    matrix_string = ["\\nabla^2 f(\\textbf{x}_0) ", "0", "Q", "\\nabla f(\\textbf{x}_0)", "J(\\textbf{x}_0)^T\\textbf{y}_0"]
    for i in range(len(matrix_list)):
        if i == 1:
            for j in range(len(g)):
                g_subs = sympy.hessian(g[j], X).subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
                latex_matrix("\\nabla^2 g_" + str(j+1) + " (\\textbf{x}_0) ", g_subs, True, col6, col7)
        else:
            subss = matrix_list[i].subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
            latex_matrix(matrix_string[i], subss, True, col6, col7)
    b_eval = b.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    g_eval = g.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    m_eval = m.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    latex_matrix_sum("\\textbf{b}-\\textbf{g}-\\textbf{m}", b_eval, g_eval, m_eval)
    LHS_subs = LHS.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    st.write(
        "We eventually get (15.14) numerically, with the coefficient matrix on the left and the right hand side (RHS) on the right.")
    latex_matrix("\\text{Coefficient Matrix}", LHS_subs, False, col6, col7)
    RHS_subs = RHS.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
    latex_matrix("RHS", RHS_subs, False, col6, col7)
    st.write("The solution to this is:")
    solv_temp = LHS_subs.LUsolve(RHS_subs)
    if option == 1:
        st.latex(sympy.latex(sympy.Matrix(["d_1^x", "d_2^x", "d_1^y", "d_2^y"])) + "= " + sympy.latex(solv_temp))
    else:
        st.latex(sympy.latex(sympy.Matrix(["d^x", "d_y"])) + "= " + sympy.latex(solv_temp))

if st.button(f"(Advanced) Show numeric steps for all remaining {k-1} iterations."):
    df1 = df.drop(columns = ['k', '||d||', 'lambda', 'f(x)'])
    for index, df_row in df1.iterrows():
        mu_value = df_row[0]
        point = list(df_row)
        st.latex(f"\\text{{We solve (15.14) numerically at the next point, }} (\\textbf{{x}}_{index+1}, \\textbf{{y}}_{index+1}).")
        #col4, col5 = st.beta_columns(2)
        #col_help = 0
        col8, col9 = st.beta_columns(2)

        col_help = 0
        matrix_list = [H, 1, Q, gradient(f, X), J.T * Y]
        matrix_string = [f"\\nabla^2 f(\\textbf{{x}}_{index+1}) ", "0", "Q", f"\\nabla f(\\textbf{{x}}_{index+1})",
                         f"J(\\textbf{{x}}_{index+1})^T\\textbf{{y}}_{index+1}"]
        for i in range(len(matrix_list)):
            if i == 1:
                for j in range(len(g)):
                    g_subs = sympy.hessian(g[j], X).subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
                    latex_matrix("\\nabla^2 g_" + str(j + 1) + f" (\\textbf{{x}}_{index+1}) ", g_subs, True, col8, col9)
            else:
                subss = matrix_list[i].subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
                latex_matrix(matrix_string[i], subss, True, col8, col9)
        b_eval = b.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
        g_eval = g.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
        m_eval = m.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
        latex_matrix_sum("\\textbf{b}-\\textbf{g}-\\textbf{m}", b_eval, g_eval, m_eval)
        LHS_subs = LHS.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
        st.write("We eventually get (15.14) numerically, with the coefficient matrix on the left and the right hand side (RHS) on the right.")
        latex_matrix("\\text{Coefficient Matrix}", LHS_subs, False, col8, col9)
        RHS_subs = RHS.subs([*zip(all_vars, point), (mu, mu_value)]).evalf()
        latex_matrix("\\text{RHS}", RHS_subs, False, col8, col9)
        st.write("The solution to this is:")
        solv_temp = LHS_subs.LUsolve(RHS_subs)
        st.latex(sympy.latex(solv_temp))





#xspace = np.linspace(-5, 5, 200)
#yspace = np.linspace(-5, 5, 200)
#Xmesh, Ymesh = np.meshgrid(xspace, yspace)
#Z = f.subs(np.vstack([Xmesh.ravel(), Ymesh.ravel()])).evalf().reshape((200,200))
#plt.pyplot.contour(X, Y, Z)



