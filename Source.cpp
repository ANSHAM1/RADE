import AutoDiff;
import Matrix;

import <iostream>;
import <memory>;
import <vector>;
using namespace std;
using namespace std_Matrix;

int main() {
	Matrix Target = Matrix(1, 3, 0);
	Target.set(0, 0, 1);

	shared_ptr<Node> X1 = make_shared<Node>(Matrix(1, 5, "random"));
	shared_ptr<Node> W1 = make_shared<Node>(Matrix(5, 3, "random"));
	shared_ptr<Node> H1 = make_shared<Node>(Matrix(1, 6, "random"));
	shared_ptr<Node> U1 = make_shared<Node>(Matrix(6, 3, "random"));
	shared_ptr<Node> B1 = make_shared<Node>(Matrix(1, 3, 0));

	shared_ptr<Node> O1 = relu(X1 * W1 + H1 * U1 + B1);

	shared_ptr<Node> X2 = make_shared<Node>(Matrix(1, 9, "random"));
	shared_ptr<Node> W2 = make_shared<Node>(Matrix(9, 3, "random"));
	shared_ptr<Node> H2 = make_shared<Node>(Matrix(1, 4, "random"));
	shared_ptr<Node> U2 = make_shared<Node>(Matrix(4, 3, "random"));
	shared_ptr<Node> B2 = make_shared<Node>(Matrix(1, 3, 0));

	shared_ptr<Node> O2 = relu(X2 * W2 + H2 * U2 + B2);
	shared_ptr<Node> OUT = O1 + O2;

	auto output = Softmaxed_CCE(OUT, Target);

	cout << "Loss : " << output.first << endl;
	cout << "Softmax : " << output.second << endl;
	return 0;
}

