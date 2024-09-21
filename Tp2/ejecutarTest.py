import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt

def generate_matrixTest(eigenValues):
    eigenvals = sorted(np.random.rand(eigenValues)*10, reverse = True)
    D = np.diag(eigenvals)
    v = np.ones((D.shape[0], 1))
    v = v / np.linalg.norm(v)
    B = np.eye(D.shape[0]) - 2 * np.outer(v, v)
    M = B.T @ D @ B
    return M, eigenvals

def guardarMatriz_file(matrices_path, filename, matrix, eigenvalues):
    filepath = os.path.join(matrices_path, filename)
    rows, cols = matrix.shape
    with open(filepath, 'w') as f:
        f.write(f"{rows} {cols}\n")
        f.write(f"{eigenvalues}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

def ejecutar(matrices_path, matrix_name, niter, autovalores, autovectores, iteraciones):
    eigen_path = '../eigen-3.4.0/eigen-3.4.0'
    project_dir = matrices_path


    cpp_file = 'main.cpp'
    executable = 'main'
    os.chdir(project_dir)

    compile_command = ['g++', '-O3', cpp_file, '-o', executable, '-I', eigen_path]
    compile_process = subprocess.run(compile_command, check=True, capture_output=True, text=True)
    

    param1 = os.path.join(matrices_path, matrix_name)
    param2 = str(niter)
    param3 = '1e-7'
    param4 = os.path.join(matrices_path, autovalores)
    param5 = os.path.join(matrices_path, autovectores)
    param6 = os.path.join(matrices_path, iteraciones)

    execute_command = ['./' + executable, param1, param2, param3, param4, param5, param6]
    execute_process = subprocess.run(execute_command, check=True, capture_output=True, text=True)
    

    if execute_process.stderr:
        print("Errores:")
        print(execute_process.stderr)
        