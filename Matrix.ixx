export module Matrix;

import <vector>;
import <iostream>;
import <random>;
import <cmath>;
using namespace std;

static std::mt19937 rng(random_device{}());

static double random_uniform(double lower, double upper) {
	uniform_real_distribution<double> dist(lower, upper);
	return dist(rng);
}

export struct Matrix {
	int ROW_SIZE;
	int COLUMN_SIZE;
	std::vector<std::vector<double>> MATRIX;

	Matrix() : ROW_SIZE(0), COLUMN_SIZE(0) { MATRIX.reserve(0); }

	Matrix(int row, int col, double val) : ROW_SIZE(row), COLUMN_SIZE(col) {
		for (size_t i = 0; i < row; i++) {
			MATRIX.push_back(std::vector<double>(col, val));
		}
	}

	Matrix(int row, int col, std::string val) : ROW_SIZE(row), COLUMN_SIZE(col) {
		if (val != "random") throw std::runtime_error("no such operator , use 'random' operator for random matrix");
		double limit = sqrt(6.0 / (row + col));
		for (size_t i = 0; i < row; i++) {
			std::vector<double> temp;
			temp.reserve(col);
			for (size_t j = 0; j < col; j++) {
				temp.push_back(random_uniform(-limit, limit));
			}
			MATRIX.push_back(temp);
		}
	}

	Matrix(std::vector<std::vector<double>> matrix) {
		ROW_SIZE = matrix.size();
		COLUMN_SIZE = matrix[0].size();
		MATRIX = matrix;
	}

	Matrix(std::vector<std::vector<char>> matrix) {
		ROW_SIZE = matrix.size();
		COLUMN_SIZE = matrix[0].size();

		std::vector<std::vector<double>> ascii;
		ascii.reserve(matrix.size());

		for (int i = 0; i < matrix.size(); i++) {
			std::vector<double> temp;
			temp.reserve(matrix[i].size());

			for (int j = 0; j < matrix[i].size(); j++) {
				temp.push_back(static_cast<double>(matrix[i][j]));
			}

			ascii.push_back(temp);
		}
		MATRIX = ascii;
	}

	double get(int i, int j) const {
		return MATRIX[i][j];
	}

	void set(int i, int j, double value) {
		MATRIX[i][j] = value;
	}

	Matrix Transpose() const {
		std::vector<std::vector<double>> transpose(COLUMN_SIZE, std::vector<double>(ROW_SIZE));
		for (int i = 0; i < ROW_SIZE; ++i) {
			for (int j = 0; j < COLUMN_SIZE; ++j) {
				transpose[j][i] = MATRIX[i][j];
			}
		}
		Matrix result(COLUMN_SIZE, ROW_SIZE, 0);
		result.MATRIX = transpose;
		return result;
	}

	Matrix reluMat() const {
		Matrix result(ROW_SIZE, COLUMN_SIZE, 0);
		for (int i = 0; i < ROW_SIZE; ++i) {
			for (int j = 0; j < COLUMN_SIZE; ++j) {
				double x = MATRIX[i][j];
				result.MATRIX[i][j] = (x > 0) ? x : 0;
			}
		}
		return result;
	}

	Matrix dReluMat() const {
		Matrix result(ROW_SIZE, COLUMN_SIZE, 0);
		for (int i = 0; i < ROW_SIZE; ++i) {
			for (int j = 0; j < COLUMN_SIZE; ++j) {
				double x = MATRIX[i][j];
				result.MATRIX[i][j] = (x > 0) ? 1.0 : 0.0;
			}
		}
		return result;
	}

	Matrix sigmoidMat() const {
		Matrix result(ROW_SIZE, COLUMN_SIZE, 0);
		for (int i = 0; i < ROW_SIZE; ++i) {
			for (int j = 0; j < COLUMN_SIZE; ++j) {
				double x = MATRIX[i][j];
				result.MATRIX[i][j] = 1.0 / (1.0 + exp(-x));
			}
		}
		return result;
	}

	Matrix dSigmoidMat() const {
		Matrix result(ROW_SIZE, COLUMN_SIZE, 0);
		for (int i = 0; i < ROW_SIZE; ++i) {
			for (int j = 0; j < COLUMN_SIZE; ++j) {
				double x = MATRIX[i][j];
				double z = 1.0 / (1.0 + exp(-x));
				result.MATRIX[i][j] = z * (1.0 - z);
			}
		}
		return result;
	}

	Matrix tanhMat() const {
		Matrix result(ROW_SIZE, COLUMN_SIZE, 0);
		for (int i = 0; i < ROW_SIZE; ++i) {
			for (int j = 0; j < COLUMN_SIZE; ++j) {
				result.MATRIX[i][j] = tanh(MATRIX[i][j]);
			}
		}
		return result;
	}

	Matrix dTanhMat() const {
		Matrix result(ROW_SIZE, COLUMN_SIZE, 0);
		for (int i = 0; i < ROW_SIZE; ++i) {
			for (int j = 0; j < COLUMN_SIZE; ++j) {
				double t = tanh(MATRIX[i][j]);
				result.MATRIX[i][j] = 1 - t * t;
			}
		}
		return result;
	}
};

export Matrix operator+(const Matrix& A, const Matrix& B) {
	if (A.ROW_SIZE != B.ROW_SIZE || A.COLUMN_SIZE != B.COLUMN_SIZE)
		throw std::runtime_error("Addition: Dimension mismatch");

	Matrix result(A.ROW_SIZE, A.COLUMN_SIZE, 0);
	for (int i = 0; i < A.ROW_SIZE; ++i)
		for (int j = 0; j < A.COLUMN_SIZE; ++j)
			result.set(i, j, A.get(i, j) + B.get(i, j));

	return result;
}

export Matrix operator+(const Matrix& A, double scalar) {
	Matrix result(A.ROW_SIZE, A.COLUMN_SIZE, 0);
	for (int i = 0; i < A.ROW_SIZE; ++i)
		for (int j = 0; j < A.COLUMN_SIZE; ++j)
			result.set(i, j, A.get(i, j) + scalar);

	return result;
}

export Matrix operator-(const Matrix& A, const Matrix& B) {
	if (A.ROW_SIZE != B.ROW_SIZE || A.COLUMN_SIZE != B.COLUMN_SIZE)
		throw std::runtime_error("Subtraction: Dimension mismatch");

	Matrix result(A.ROW_SIZE, A.COLUMN_SIZE, 0);
	for (int i = 0; i < A.ROW_SIZE; ++i)
		for (int j = 0; j < A.COLUMN_SIZE; ++j)
			result.set(i, j, A.get(i, j) - B.get(i, j));

	return result;
}

export Matrix operator-(const Matrix& A, double scalar) {
	Matrix result(A.ROW_SIZE, A.COLUMN_SIZE, 0);
	for (int i = 0; i < A.ROW_SIZE; ++i)
		for (int j = 0; j < A.COLUMN_SIZE; ++j)
			result.set(i, j, A.get(i, j) - scalar);

	return result;
}

export Matrix operator*(const Matrix& A, const Matrix& B) {
	if (A.COLUMN_SIZE != B.ROW_SIZE)
		throw std::runtime_error("Multiplication: Invalid dimensions");

	Matrix result(A.ROW_SIZE, B.COLUMN_SIZE, 0);

	for (int i = 0; i < A.ROW_SIZE; ++i)
		for (int j = 0; j < B.COLUMN_SIZE; ++j)
			for (int k = 0; k < A.COLUMN_SIZE; ++k)
				result.set(i, j, result.get(i, j) + A.get(i, k) * B.get(k, j));

	return result;
}

export Matrix operator*(const Matrix& A, double scalar) {
	Matrix result(A.ROW_SIZE, A.COLUMN_SIZE, 0);
	for (int i = 0; i < A.ROW_SIZE; ++i)
		for (int j = 0; j < A.COLUMN_SIZE; ++j)
			result.set(i, j, A.get(i, j) * scalar);

	return result;
}

export Matrix operator*(double scalar, const Matrix& A) {
	return A * scalar;
}

export Matrix operator/(const Matrix& A, double scalar) {
	if (scalar == 0.0)
		throw std::runtime_error("Division by zero");

	Matrix result(A.ROW_SIZE, A.COLUMN_SIZE, 0);
	for (int i = 0; i < A.ROW_SIZE; ++i)
		for (int j = 0; j < A.COLUMN_SIZE; ++j)
			result.set(i, j, A.get(i, j) / scalar);

	return result;
}

export namespace std_Matrix {
	export Matrix outerProduct(const Matrix& A, const Matrix& B) {
		if (A.COLUMN_SIZE != 1 && A.ROW_SIZE != 1)
			throw std::runtime_error("First argument is not a vector (must be row or column vector)");
		if (B.COLUMN_SIZE != 1 && B.ROW_SIZE != 1)
			throw std::runtime_error("Second argument is not a vector (must be row or column vector)");

		std::vector<double> vecA, vecB;

		if (A.COLUMN_SIZE == 1) {
			for (int i = 0; i < A.ROW_SIZE; ++i)
				vecA.push_back(A.get(i, 0));
		}
		else {
			for (int j = 0; j < A.COLUMN_SIZE; ++j)
				vecA.push_back(A.get(0, j));
		}

		if (B.COLUMN_SIZE == 1) {
			for (int i = 0; i < B.ROW_SIZE; ++i)
				vecB.push_back(B.get(i, 0));
		}
		else {
			for (int j = 0; j < B.COLUMN_SIZE; ++j)
				vecB.push_back(B.get(0, j));
		}

		Matrix result(vecA.size(), vecB.size(), 0);
		for (int i = 0; i < vecA.size(); ++i)
			for (int j = 0; j < vecB.size(); ++j)
				result.set(i, j, vecA[i] * vecB[j]);

		return result;
	}

	export Matrix elementaryProduct(const Matrix& A, const Matrix& B) {
		if (A.ROW_SIZE != B.ROW_SIZE || A.COLUMN_SIZE != B.COLUMN_SIZE)
			throw std::runtime_error("Elementwise product not possible: Size mismatch");

		Matrix result(A.ROW_SIZE, A.COLUMN_SIZE, 0);
		for (int i = 0; i < A.ROW_SIZE; ++i)
			for (int j = 0; j < A.COLUMN_SIZE; ++j)
				result.set(i, j, A.get(i, j) * B.get(i, j));

		return result;
	}

	export bool operator==(const Matrix& A, const Matrix& B) {
		if (A.ROW_SIZE != B.ROW_SIZE || A.COLUMN_SIZE != B.COLUMN_SIZE) return false;
		for (int i = 0; i < A.ROW_SIZE; ++i)
			for (int j = 0; j < A.COLUMN_SIZE; ++j) {
				if (A.get(i, j) != B.get(i, j)) return false;
			}
		return true;
	}

	export bool operator!=(const Matrix& A, const Matrix& B) {
		return !(operator==(A, B));
	}

	export std::ostream& operator<<(std::ostream& os, const Matrix& A) {
		for (int i = 0; i < A.ROW_SIZE; ++i) {
			for (int j = 0; j < A.COLUMN_SIZE; ++j) {
				os << A.get(i, j) << ' ';
			}
			os << '\n';
		}
		return os;
	}
}

export namespace std_Loss {

	export double MSE(const Matrix& prediction, const Matrix& target) {
		double sum = 0.0;
		for (int i = 0; i < prediction.ROW_SIZE; ++i)
			for (int j = 0; j < prediction.COLUMN_SIZE; ++j)
				sum += pow(prediction.get(i, j) - target.get(i, j), 2);
		return sum / (prediction.ROW_SIZE * prediction.COLUMN_SIZE);
	}

	export Matrix dMSE(const Matrix& prediction, const Matrix& target) {
		Matrix grad(prediction.ROW_SIZE, prediction.COLUMN_SIZE, 0);
		double scale = 2.0 / (prediction.ROW_SIZE * prediction.COLUMN_SIZE);

		for (int i = 0; i < prediction.ROW_SIZE; ++i)
			for (int j = 0; j < prediction.COLUMN_SIZE; ++j)
				grad.set(i, j, scale * (prediction.get(i, j) - target.get(i, j)));

		return grad;
	}

	export double BCE(const Matrix& prediction, const Matrix& target) {
		double sum = 0.0;
		for (int i = 0; i < prediction.ROW_SIZE; ++i)
			for (int j = 0; j < prediction.COLUMN_SIZE; ++j) {
				double p = prediction.get(i, j);
				double t = target.get(i, j);
				sum += -(t * log(p + 1e-9) + (1 - t) * log(1 - p + 1e-9));
			}
		return sum / (prediction.ROW_SIZE * prediction.COLUMN_SIZE);
	}

	export Matrix dBCE(const Matrix& prediction, const Matrix& target) {
		Matrix grad(prediction.ROW_SIZE, prediction.COLUMN_SIZE, 0);
		double scale = 1.0 / (prediction.ROW_SIZE * prediction.COLUMN_SIZE);

		for (int i = 0; i < prediction.ROW_SIZE; ++i)
			for (int j = 0; j < prediction.COLUMN_SIZE; ++j) {
				double p = prediction.get(i, j);
				double t = target.get(i, j);
				grad.set(i, j, scale * (p - t) / ((p + 1e-9) * (1.0 - p + 1e-9)));
			}

		return grad;
	}

	export double CCE(const Matrix& prediction, const Matrix& target) {
		double sum = 0.0;
		for (int i = 0; i < prediction.ROW_SIZE; ++i)
			for (int j = 0; j < prediction.COLUMN_SIZE; ++j)
				sum += -target.get(i, j) * log(prediction.get(i, j) + 1e-9);
		return sum / prediction.ROW_SIZE;
	}

	export Matrix dCCE(const Matrix& prediction, const Matrix& target) {
		Matrix grad(prediction.ROW_SIZE, prediction.COLUMN_SIZE, 0);
		double scale = -1.0 / prediction.ROW_SIZE;

		for (int i = 0; i < prediction.ROW_SIZE; ++i)
			for (int j = 0; j < prediction.COLUMN_SIZE; ++j) {
				double p = prediction.get(i, j);
				double t = target.get(i, j);
				grad.set(i, j, scale * t / (p + 1e-9));
			}

		return grad;
	}

	export Matrix dSoft_CCE(const Matrix& prediction, const Matrix& target) {
		Matrix grad(prediction.ROW_SIZE, prediction.COLUMN_SIZE, 0);

		for (int i = 0; i < prediction.ROW_SIZE; ++i)
			for (int j = 0; j < prediction.COLUMN_SIZE; ++j) {
				double p = prediction.get(i, j);
				double t = target.get(i, j);
				grad.set(i, j, p - t);
			}

		return grad;
	}
}

export Matrix softmax(Matrix& M) {
	Matrix result = Matrix(M.ROW_SIZE, M.COLUMN_SIZE, 0);
	for (int i = 0; i < M.ROW_SIZE; ++i) {
		double maxVal = *max_element(M.MATRIX[i].begin(), M.MATRIX[i].end());
		double sum = 0.0;

		for (int j = 0; j < M.COLUMN_SIZE; ++j) {
			result.set(i, j, exp(M.get(i, j) - maxVal));
			sum += M.get(i,j);
		}
		for (int j = 0; j < M.COLUMN_SIZE; ++j)
			result.set(i, j, M.get(i, j) / sum);
	}
	return result;
}