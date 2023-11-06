#include "parameter.h"
#include "function.h"

extern int     neighbors[number_of_computers][number_of_neighbors];
extern double  prob[number_of_states][number_of_states][number_of_states][number_of_states];
NumArray3D     weight;
int type = 3;   // default: single-direct ring
// type 1:  bi-direct ring
// type 2:  star
// type 3:  ring and star
// type 4:  rings and rings
// type 5:  3 legs

int             state_constr[number_of_computers][number_of_constraints];
int             action_constr[number_of_computers][number_of_constraints];
int             constr_num = 0;

int main(int argc, const char* argv[]) {

    create_tran_prob();
    create_network_topology(type);
    create_basis_functions(type);

    GRBEnv* env = new GRBEnv();

    GRBModel model = GRBModel(env);

    NumVar3D alpha;

    creat_mastered_problem_NonRobust(model, alpha);

    double eps = 0.00001, value = 0;
    for (int iter = 0; iter < 100 * number_of_computers; iter++) {
        for (int inner_iter = 0; inner_iter < 5; inner_iter++) {
            random_generate_constraints_local_search_NonRobust(model, alpha, weight, 2 * number_of_computers, constr_num);
            model.optimize();

            for (int i = 0; i < number_of_states; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    for (int k = 0; k < number_of_computers; k++) {
                        weight[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
                    }
                }
            }
        }

        cout << "number of constraints:" << model.get(GRB_IntAttr_NumConstrs) << "  ";
        cout << "value of master problem: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

        if ((model.get(GRB_DoubleAttr_ObjVal) < (1 + eps) * value) && (iter > number_of_computers)) {
            break;
        }

        value = model.get(GRB_DoubleAttr_ObjVal);
    }

    double value_all_one = 0;
    int state[number_of_computers];
    for (int k = 0; k < number_of_computers; k++) {
        value_all_one += weight[1][1][k];
        state[k] = 1;
    }

    double value_mc_non = monto_carlo_MIP(weight, state, value_all_one);

    delete env;

    GRBEnv* env_true = new GRBEnv();

    GRBModel model_true = GRBModel(env_true);

    creat_mastered_problem_NonRobust(model_true, alpha);

    value = 0;
    for (int iter = 0; iter < 100 * number_of_computers; iter++) {
        for (int inner_iter = 0; inner_iter < 5; inner_iter++) {
            random_generate_constraints_local_search_true(model_true, alpha, weight, 2 * number_of_computers);
            model_true.optimize();

            for (int i = 0; i < number_of_states; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    for (int k = 0; k < number_of_computers; k++) {
                        weight[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
                    }
                }
            }
        }

        cout << "number of constraints:" << model_true.get(GRB_IntAttr_NumConstrs) << "  ";
        cout << "value of master problem: " << model_true.get(GRB_DoubleAttr_ObjVal) << endl;

        if ((model_true.get(GRB_DoubleAttr_ObjVal) < (1 + eps) * value) && (iter > number_of_computers)) {
            break;
        }

        value = model_true.get(GRB_DoubleAttr_ObjVal);
    }

    double value_mc_true = monto_carlo_MIP_true(weight, state, value_all_one) - value_mc_non;

    delete env_true;

    double value_old = 0, value_new;
    for (int iter = 0; iter < 10; iter++) {
        cout << "round: " << iter << endl;
        random_generate_constraints_local_search(weight, 300, constr_num);
        if (constr_num >= number_of_constraints) break;
        value_new = mastered_problem_weight(constr_num, weight);
        if ((value_new > 0) && (value_new < value_old * 1.1)) break;
        else value_old = value_new;
        cout << "master problem value : " << value_old << endl;
        for (int i = 1; i < 10; i++) {
            value_new = mastered_problem_weight(constr_num, weight);
            cout << "master problem value : " << value_new << endl;
            if (value_new > 0.9 * value_old) {
                value_old = value_new;
                break;
            }
            else value_old = value_new;
            if (value_new < 0) break;
        }

    }



    cout << endl << "Type of topology: " << type << "; number of computers: " << number_of_computers << endl << endl;

    cout << endl << "value for all one state:" << value_all_one << endl;

    double value_mc = monto_carlo_MIP(weight, state, value_all_one) - value_mc_true - value_mc_non;

    cout << endl << endl;

    cout << "Non Robust case Monte Carlo value: " << value_mc_non << endl;

    cout << "Robust case Monte Carlo value: " << value_mc << endl;

    cout << "True case Monte Carlo value: " << value_mc_true << endl;

    return 0;

}