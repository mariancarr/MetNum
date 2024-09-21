Para ejecutar en windows:
Modificar en ejecutarTests.py la variable "eigen_path" de la función ejecutar con la ruta donde se encuentra instalado eigen.

Para ejecutar en linux:
En ejecutarTest.py:
	En la función "ejecutar" cambiar "compile_command", "compile_process", "execute_command", "execute_process" por:
    	compile_command = ['g++', '-O3', cpp_file, '-o', executable]
        compile_process = subprocess.run("g++ -O3 main.cpp -o main", check=True, capture_output=True, text=True,shell=True)

        execute_command = f"./{executable} {param1} {param2} {param3} {param4} {param5} {param6}"
        execute_process = subprocess.run(execute_command, check=True, capture_output=True, text=True,shell=True)
