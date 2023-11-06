
#ifndef FUNCTION_H
#define FUNCTION_H

int decr(int i);

int incr(int i);

void create_tran_prob();

void create_basis_functions(int type);

void create_network_topology(int type);

double mastered_problem_weight(int constr_num, NumArray3D& weight);

double transprob_lp(NumArray3D& weight, NumArray3D& prob_sa, const int* current_state,
    const int* current_action);

void generate_state_action(int* current_state, int* current_action);

void random_generate_constraints_local_search(NumArray3D& weight, int number_of_constr, int& constr_num);

double monto_carlo_MIP(NumArray3D weight, const int* initial_state, const double upper_bound);

void creat_mastered_problem_NonRobust(GRBModel& model, NumVar3D& alpha);

double transprob_lp(NumArray3D& weight, NumArray3D& prob_sa, const int* current_state,
    const int* current_action);

void random_generate_constraints_local_search_NonRobust(GRBModel& model, NumVar3D& alpha, NumArray3D& weight,
    int number_of_constr, int& constr_num);

void random_generate_constraints_local_search_true(GRBModel& model, NumVar3D& alpha, NumArray3D& weight,
    int number_of_constr);

void add_constraint_to_master_problem(GRBModel& model, NumVar3D& alpha,
    const int* current_state, const int* current_action, int& constr_num);

void add_constraint_to_master_problem_true(GRBModel& model, NumVar3D& alpha,
    const int* current_state, const int* current_action);

double prob_out(int* current_state, int* original_state, int i, int action_i, int state_i, int current_time);

double monto_carlo_MIP_true(NumArray3D weight, const int* initial_state, const double upper_bound);

double monto_carlo_MIP_robust(NumArray3D& weight, const int* initial_state, const double upper_bound);
#endif

