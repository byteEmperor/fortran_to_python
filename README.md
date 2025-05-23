# fortran_to_python
help a friend with legacy code

## How to change the derivs function?
1) add a new derivs python file in the models folder
2) make sure that it uses the same input-output framework
3) import it into odeint_scipy.py
4) modify line 30 in odeint_scipy.py with the new derivs function
5) run it normally