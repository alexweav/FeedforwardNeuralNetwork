using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Matrices {
    public class Matrix {

        private float[,] data;
        private int numRows;
        private int numColumns;

        //constructors
        #region Constructors

        //Constructs a matrix with n rows and n columns filled with zeros
        public Matrix(int n) {
            if (n < 1) {
                throw new ArgumentException("Cannot have a " + n + "x" + n + " matrix.");
            }
            data = new float[n, n];
            numRows = n;
            numColumns = n;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    data[i, j] = 0;
                }
            }
        }

        //Constructs a matrix with m rows and n columns filled with zeros
        public Matrix(int m, int n) {
            if (m < 1 || n < 1) {
                throw new ArgumentException("Cannot have a " + m + "x" + n + " matrix.");
            }
            data = new float[m, n];
            numRows = m;
            numColumns = n;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    data[i, j] = 0;
                }
            }
        }

        //Constructs a matrix from a pre-existing 2D array
        public Matrix(float[,] data) {
            if (data == null) {
                throw new NullReferenceException("Cannot build matrix from null array.");
            }
            numRows = data.GetLength(0);
            numColumns = data.GetLength(1);
            if (numRows < 1 || numColumns < 1) {
                throw new ArgumentException("Can only build matrices from nonempty 2D arrays.");
            }
            this.data = data;
        }

        #endregion

        //gets, sets if present
        #region GetSet

        public float[,] Data {
            get {
                return data;
            }
        }

        public int NumRows {
            get {
                return numRows;
            }
        }

        public int NumColumns {
            get {
                return numColumns;
            }
        }

        #endregion

        //Indices, equality, general use methods
        #region Behavior

        //Matrices are indexed using a 1-based system, as notated mathematically
        public float this[int i, int j] {
            get {
                return data[i - 1, j - 1];
            }
            set {
                data[i - 1, j - 1] = value;
            }
        }

        //Two matrices are considered equal if and only if they have the same dimensions and 
        //every entry a[i,j] of the first matrix equals every entry b[i,j] of the second matrix
        public override bool Equals(object obj) {
            if (obj == null) {
                return false;
            }
            Matrix m = obj as Matrix;
            if ((object)m == null) {
                return false;
            }
            if (this.NumRows != m.NumRows || this.NumColumns != m.NumColumns) {
                return false;
            }
            return AreEqual2DArrays(this.Data, m.Data);
        }

        //Equals() again, but takes another matrix rather than any object
        public bool Equals(Matrix m) {
            if ((object)m == null) {
                return false;
            }
            if (this.NumRows != m.NumRows || this.NumColumns != m.NumColumns) {
                return false;
            }
            return AreEqual2DArrays(this.Data, m.Data);
        }

        //The equality statement between matrices a and b can be notated as "a == b"
        public static bool operator ==(Matrix a, Matrix b) {
            if (System.Object.ReferenceEquals(a, b)) {
                return true;    //return true if they are literally the same object
            }

            if (((object)a == null) || ((object)b == null)) {
                return false;   //return false if one or the other is null
            }
            return a.Equals(b); //defer to the normal equality definition
        }

        //The inequality statement between matrices a and b can be notated as "a != b"
        public static bool operator !=(Matrix a, Matrix b) {
            return !(a == b);
        }

        private static bool AreEqual2DArrays(float[,] a, float[,] b) {
            if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1)) {
                return false;
            }
            for (int i = 0; i < a.GetLength(0); ++i) {
                for (int j = 0; j < a.GetLength(1); ++j) {
                    if (a[i, j] != b[i, j]) {
                        return false;
                    }
                }
            }
            return true;
        }

        #endregion

        //Operations on matrices
        #region Operations

        //Adds two matrices and returns the result.  The matrices must have the same dimensions in order to add.
        public static Matrix Add(Matrix a, Matrix b) {
            if (a == null || b == null) {
                throw new NullReferenceException("Null matrices cannot be added.");
            }
            if (a.NumRows != b.NumRows || a.NumColumns != b.NumColumns) {
                throw new ArgumentException("Matrices must have the same dimensions to be added.");
            }
            Matrix result = new Matrix(a.NumRows, a.NumColumns);
            for (int i = 1; i <= a.NumRows; ++i) {
                for (int j = 1; j <= a.NumColumns; ++j) {
                    result[i, j] = a[i, j] + b[i, j];
                }
            }
            return result;
        }

        //Addition of two matrices a and b can be expressed as "a + b"
        public static Matrix operator +(Matrix a, Matrix b) {
            return Add(a, b);
        }

        //Multiplies every entry in the given matrix by a constant c and returns the new matrix
        //Also known as "scalar multiplication"
        public static Matrix Multiply(Matrix a, float c) {
            if (a == null) {
                throw new NullReferenceException("Cannot multiply by null matrix.");
            }
            Matrix result = new Matrix(a.NumRows, a.NumColumns);
            for (int i = 1; i <= a.NumRows; ++i) {
                for (int j = 1; j <= a.NumColumns; ++j) {
                    result[i, j] = a[i, j] * c;
                }
            }
            return result;
        }

        //Scalar multiplication is commutative
        public static Matrix Multiply(float c, Matrix a) {
            return Multiply(a, c);
        }

        //Scalar multiplication of a matrix a and a constant c may be written as "a * c"
        public static Matrix operator *(Matrix a, float c) {
            return Multiply(a, c);
        }

        //Multiplication of a matrix a and constant c may also be written as "c * a" due to commutativity
        //The type difference requires a separate definition
        public static Matrix operator *(float c, Matrix a) {
            return Multiply(c, a);
        }

        //Multiplies two matrices and returns the result matrix
        //If the first matrix is m x n and the second matrix is n x p, then the result matrix is m x p
        //The given matrices must share the appropriate dimension in order to be multiplied
        public static Matrix Multiply(Matrix a, Matrix b) {
            if (a == null || b == null) {
                throw new NullReferenceException("Cannot multiply by null matrix.");
            }
            if (a.NumColumns != b.NumRows) {
                throw new ArgumentException("In order to multiply matrices A and B, the number of columns of A must equal the number of rows of B");
            }
            Matrix output = new Matrix(a.NumRows, b.NumColumns);
            for (int i = 1; i <= a.NumRows; ++i) {
                for (int j = 1; j <= b.NumColumns; ++j) {
                    float sum = 0;
                    for (int k = 1; k <= a.NumColumns; ++k) {
                        sum += a[i, k] * b[k, j];
                    }
                    output[i, j] = sum;
                }
            }
            return output;
        }

        //Multiplication of two matrices a and b may be written as "a * b"
        public static Matrix operator *(Matrix a, Matrix b) {
            return Multiply(a, b);
        }

        //The negative of a matrix a, denoted "-a", is given by multiplying the matrix a by -1.
        public static Matrix operator -(Matrix a) {
            return Multiply(-1, a);
        }

        //Subtraction between matrices a and b, denoted "a - b", is defined as follows:
        //      a - b := a + (-b)
        public static Matrix operator -(Matrix a, Matrix b) {
            return Add(a, -b);
        }

        //For two matrices a and b of the same dimensions, the Hadamard product is a matrix for which
        //each element is the product of its two corresponding elements in matrices a and b
        //HadamardProduct(a, b)[i, j] = a[i, j] * b[i, j]
        public static Matrix HadamardProduct(Matrix a, Matrix b) {
            if (a == null || b == null) {
                throw new NullReferenceException("Cannot take the Hadamard product of null matrices.");
            }
            if (a.NumColumns != b.NumColumns || a.NumRows != b.NumRows) {
                throw new ArgumentException("Cannot take the Hadamard product of matrices of different dimensions.");
            }
            Matrix output = new Matrix(a.NumRows, a.NumColumns);
            for (int i = 1; i <= a.NumRows; ++i) {
                for (int j = 1; j <= a.NumColumns; ++j) {
                    output[i, j] = a[i, j] * b[i, j];
                }
            }
            return output;
        }

        //Given a matrix a, the transpose of a is the matrix in which the ith row, jth column element of Transpose(a) is the jth row, ith column element of a.
        //i.e., Transpose(a)[i, j] = a[j, i]
        //This is the same as reflecting the matrix a over its main diagonal.
        public static Matrix Transpose(Matrix a) {
            if (a == null) {
                throw new NullReferenceException("Cannot take the transpose of a null matrix.");
            }
            Matrix output = new Matrix(a.NumColumns, a.NumRows);
            for (int i = 1; i <= output.NumRows; ++i) {
                for (int j = 1; j <= output.NumColumns; ++j) {
                    output[i, j] = a[j, i];
                }
            }
            return output;
        }
        #endregion
    }
}
