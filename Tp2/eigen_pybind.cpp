#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

using namespace std;
using namespace Eigen;
namespace py=pybind11;

typedef tuple<double, VectorXd, int> metodoPot_res;
typedef tuple<VectorXd, MatrixXd, vector<int>> eigenRes;

metodoPot_res metodoPotencia( const MatrixXd &A, int niter, double eps) {
    long n = A.cols(); //asumo q es cuadrada
    VectorXd v = VectorXd::Random(n);
    double lambda;
    int i = 0;
    double y = 1;
    while((i < niter) && (y > eps)){
        y = 1;

        //vector Asociado
        VectorXd aux = A * v;
        v = aux / aux.norm();

        //AutoValor "ESTO SE PODRIA HACER AFUERA DEL WHILE SI SOLO NOS QUEREMOS QUEDAR CON EL ULTITMO, LO DEJO ACA Y DESP VEMOS"
        lambda = v.transpose() *  A * v;
        lambda = lambda / (v.transpose() * v);


        //aca estoy buscando el error para ver si y <= a eps
        VectorXd yAux = A * v;
        yAux = yAux - (v * lambda);
        y = yAux.norm();

        i++;
    }

//devuelvo el i para ver la cantidad de iteraciones con las que sale para el 1b.
    metodoPot_res res = make_tuple(lambda, v, i);
    return res;
}

eigenRes eigen(const MatrixXd &A, int num, int niter, double eps){
    long n = A.cols();  //asumo cuadrada
    MatrixXd A_copia = A;
    VectorXd valores(num);
    MatrixXd vectores(n,num);
    vector<int> iteraciones(num,0);
    for(int i = 0; i < num; i++){
        metodoPot_res p = metodoPotencia(A_copia, niter, eps);
        double autoValor = get<0>(p);
        VectorXd autoVector = get<1>(p);
        int iteracion = get<2>(p);
        iteraciones[i] = iteracion;
        valores(i) = autoValor;
        vectores.row(i) = autoVector.transpose();
        MatrixXd aux = autoValor * (autoVector * autoVector.transpose()) ;
        A_copia = A_copia - aux;
    }

    eigenRes res = make_tuple(valores, vectores, iteraciones);
    return res;
}


PYBIND11_MODULE(metoPotencia, m) {
    m.def("metodoPotencia", &metodoPotencia, "Método de la potencia",
    py::arg("A"), pybind11::arg("niter"), pybind11::arg("eps"));
    m.def("eigen", &eigen, "Calculo de autovalores y autovectores",
    py::arg("A"), pybind11::arg("num"), pybind11::arg("niter"), pybind11::arg("eps"));
};




int main(int argc, char** argv) {
    if (argc != 7) {
        cerr << "error de parametros" << endl;
        return 1;
    }

    // matrizInput niter eps eigenvalues(salida) eigenvectors(salida) iteraciones(salida)

    const char* input = argv[1];
    int niter = atoi(argv[2]);
    double eps = atof(argv[3]);
    const char* autoValores_salida = argv[4];
    const char* autoVectores_salida = argv[5];
    const char* iteraciones_salida = argv[6];

    ifstream file(input);
    if (!file.is_open()) {
        cerr << " error al abrir archivo: " << input << endl;
        return 1;
    }

    int filas;
    int cols;
    int num;
    file >> filas >> cols >> num;
    MatrixXd A(filas, cols);
    for(int i = 0; i < filas; i++){
        for(int j = 0; j < cols; j++){
            file >> A(i,j);
        }
    }

    file.close();

    eigenRes res = eigen(A, num, niter, eps);
    MatrixXd resA = get<1>(res);
    VectorXd resLambda = get<0>(res);
    vector<int> resIteraciones = get<2>(res);

    ofstream salida1(autoValores_salida);
    if (!salida1.is_open()) {
        cerr << "error al abrir archivo: " << autoValores_salida << endl;
        return 1;
    }

    for (int i = 0; i < resLambda.size(); ++i) {
        if (i > 0) {
            salida1 << " ";
        }
        salida1 << resLambda(i);
    }
    salida1.close();


    ofstream salida2(autoVectores_salida);
    if (!salida2.is_open()) {
        cerr << "error al abrir archivo: " << autoVectores_salida << endl;
        return 1;
    }

    for (int i = 0; i < filas; i++) {
        salida2 << resA.col(i).transpose() <<endl;
    }
    salida2.close();

    ofstream salida3(iteraciones_salida);
    if (!salida3.is_open()) {
        cerr << "error al abrir archivo: " << iteraciones_salida << endl;
        return 1;
    }

    for (int i = 0; i < resIteraciones.size(); ++i) {
        if (i > 0) {
            salida3 << " ";
        }
        salida3 << resIteraciones[i];
    }
    salida3.close();

    
    return 0;
}
