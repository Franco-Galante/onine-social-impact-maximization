/***********************************************************************************************************************
 * 
 * std::map implementation of the trellis-like approach for finding the optimal solution for an influencer which would 
 * like to attract a pool of users toward the right-extreme of the interval [0,1] on a finite time horizon N. The avera-
 * ge opinion of the users is used as 'reward' metric.
 * 
 * HYPOTESIS: - users are distributed according to 2 deltas (d1, d2), s.t. x(0)=z, the prejudice equals initial opinion
 *            - |psi|=1 or 0 (0-1 function): the delta distrbutions never split, either all user move or none does
 *            - the opinion space is discretized in B intervals of with dx    
 *            - there is only ONE influencer in the system, who at each step can chose any xi in the opinion space [0,1]
 * NOTES:     - users can go back towards their prejudice when they are NOT reached by a message of the influencer      
 *            - the STATE of the system is characterized by (bin(d1), bin(d2)), where bin(.) indicates the discrete opi-
 *              nion bin in which the delta is
 *                                                      
***********************************************************************************************************************/

#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include <utility>
#include <cstring>
#include <cmath>
#include <string>
#include <iomanip>


/***************************************************** FUNCTIONS ******************************************************/

void err_and_exit(std::string err_msg)
{
    std::cout << err_msg << std::endl;
    exit(1);
}


// d  : distance between user opinion and influencer opinion (abs(x-xi))
// a,b: free parameters of the psi function
double psi(double d, double a, double b, int B)
{
    double epsilon = (1.0/(2.0*(double)B))*0.001; // smaller than discretization step

    double p = 0;
    if (a<0.0 || a>1.0 || b<0.0 || b>1.0) // parameters check
        err_and_exit("ERROR: psi parameters outside [0,1]");

    if (d < b+epsilon) p = a;
    return p;
}


// z:  prejudice of the group (delta)
// x:  current opinion of the group
// xi: opinion chosen by the influencer
double ds(double a, double b, double z, double x, double xi, bool z_back, double p1, double p2, int B)
{
    if (psi(std::abs(x-xi), p1, p2, B) == 0)
    {
        // when not reached by xi users feel an attraction towards their prejudice
        if (z_back)
        {
            // renormalize the weights to still have a convex combination
            double norm_a = a / (a+b);
            double norm_b = b / (a+b);
            return norm_a*z + (norm_b-1.0)*x;
        }
        // no attraction towards prejudice, x(t+1) = x(t) -> ds=0, no movement
        else
            return 0.0; 
    }
    else
        return a*z + (b-1.0)*x + (1.0-a-b)*xi;
}


// computes an upper bound to the maximum achievable opinion given a certain z
double upper_x(double a, double b, double z, int n_bins)
{
    double factor = 1.0 / (1.0-b);
    // assuming the influencer pulls towards 1.0
    double xi_ub = (1.0 + (double)(n_bins-1)*2.0) / (2.0*(double)n_bins);
    return factor * (a*z + (1.0-a-b)*xi_ub);
}


// from a real opinion value it gives back the index of the bin it belongs to
int x_to_idx(double x, int B_p)
{
    double dx = 1.0 / (double)B_p;
    return (int)floor(x / dx);
}


template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) // to plot vectors
{
	os << "[";
	for (unsigned i = 0; i < v.size(); ++i) {
		os << v[i];
		if (i != v.size() - 1) os << ", ";
	}
	os << "]"; // do NOT go to new line automatically
	return os;
}


// prints the tuple with fixed size equal to 3 (and with the specified types)
void print_tuple(std::tuple<int, double, double> t)
{
    std::cout << "(p:" << std::get<0>(t) << ",xi:" << std::get<1>(t) << ",r:" << std::get<2>(t) << ")";
}


// check if a double value is in the interval [0,1]
void in_01(double v, std::string v_id)
{
    if (v < 0.0 || v > 1.0)
        err_and_exit("ERROR: " + v_id + " must be in [0,1]");
}


// parse the values after an option from command line (option check done by the calling function)
double double_arg(int& i, int argc, char**& argv)
{
    if (i + 1 < argc)
    {
        // take the next passed value after the option and compare only the first element
        if (strncmp(argv[i+1], "-", 1) == 0)
            err_and_exit("ERROR: a value must be passed after an option");

        i++;                       // jump to next argument
        return std::stod(argv[i]); // convert to double
    }
    else {
        err_and_exit("ERROR: a value must be passed after an option -v");
        return -1.0; // never reached
    }
}

int int_arg(int& i, int argc, char**& argv)
{
    if (i + 1 < argc)
    {
        if (strncmp(argv[i+1], "-", 1) == 0)
            err_and_exit("ERROR: a value must be passed after an option");

        i++;                       
        return std::stoi(argv[i]);
    }
    else {
        err_and_exit("ERROR: a value must be passed after the option -v");
        return -1;
    }
}


// when option '-h' (help) is invoked this prints the help message for the script with informations on the options
void script_info()
{
    std::cout << std::endl;
    std::cout << "********************************************************************************" << std::endl;
    std::cout << "** Script to find the optimal strategy for one influencer to pull users to 1. **" << std::endl;
    std::cout << "*                                                                              *" << std::endl;
    std::cout << "*  --help/-h     : info about the available options                            *" << std::endl;
    std::cout << "*  --recall      : (flag) if set, users  attracted toward z when not reached   *" << std::endl;
    std::cout << "*  --intervals/-i: number of intervals in which the opinion space is divided   *" << std::endl;
    std::cout << "*  --n-iter/-n   : number of iteration considered for the optimization         *" << std::endl;
    std::cout << "*  --users/-u    : number of regular users                                     *" << std::endl;
    std::cout << "*  --alpha/-a    : weight on the prejudice z in the opinion updating           *" << std::endl;
    std::cout << "*  --beta/-b     : weight on the current opinion x in the opinion updating     *" << std::endl;
    std::cout << "*  --ratio/-r    : ratio between cardinality in the 1st and in the 2nd delta   *" << std::endl;
    std::cout << "*  --xitarget    : target influencer's opinion value (desired attraction point)*" << std::endl;
    std::cout << "*  -z1           : prejudice of the first delta                                *" << std::endl;
    std::cout << "*  -z2           : prejudice of the second delta                               *" << std::endl;
    std::cout << "*  -x1           : initial opinion of the first delta                          *" << std::endl;
    std::cout << "*  -x2           : initial opinion of the second delta                         *" << std::endl;
    std::cout << "*  -p2           : width of the rect (psi) function                            *" << std::endl;
    std::cout << "*  --call        : to set when called by other script, use stdout to pass vars *" << std::endl;
    std::cout << "*  --debug       : trigger the verbose debug mode                              *" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    exit(0);
}


// returns the sequences of states (pairs of opinion values) for the optimal solution using the trellis-like graph
std::vector<int> print_states(std::vector<std::map<int, std::tuple<int, double, double>>>& g, int state, bool c)
{
    int pts = g.size(); // number of states
    int N   = pts - 1;

    std::vector<int> state_opt(pts,0);
    int prev_idx = state;              // the last state reached, which points to the previous state
    state_opt.back() = prev_idx;       // save last state (at idx N+1)

    // the info of the state is embedded in the key of the map, each state contains the previous state
    for (int i=N; i > 0; i--) {  // loop over the (reverse) time index, start from the last time (N)
        prev_idx = std::get<0>(g[i].at(prev_idx));
        state_opt[i-1] = prev_idx;
    }

    if (!c) {
        std::cout << std::endl << " **            optimal states seqence:            ** " << std::endl;
        std::cout << state_opt << std::endl;
    }
    else {
        std::cout << state_opt << std::endl; // needs to be the last method to be invoked
    }

    return state_opt; // return the optimal states, need to check that is the quickest
}


double state_to_avg_x(std::vector<std::pair<int, int>>& states, std::vector<double>& mid_bin, int state, double d1_over_d2)
{
    double x_d1 = mid_bin[states[state].first];
    double x_d2 = mid_bin[states[state].second];

    return d1_over_d2*x_d1 + (1.0-d1_over_d2)*x_d2;
}


// does the same as above but explicitly computes the average opinion of the two-user-group population
void print_opt_avg(std::vector<std::map<int, std::tuple<int, double, double>>>& g, int state, 
                   std::vector<std::pair<int, int>>& states, std::vector<double>& mid_bin, double d1_over_d2, bool c)
{
    int pts = g.size(); // number of states
    int N   = pts - 1;
    int prev_state = std::get<0>(g[N].at(state));

    std::vector<double> avg_opt(pts,0); // init with pts=N+1
    avg_opt.back() = state_to_avg_x(states, mid_bin, state, d1_over_d2);

    for (int i=N-1; i >= 0; i--)
    {
        avg_opt[i] = state_to_avg_x(states, mid_bin, prev_state, d1_over_d2); 
        prev_state = std::get<0>(g[i].at(prev_state));
    }

    if (!c)
    {
        std::cout << std::endl << " **           The average sequence is:            ** " << std::endl;
        std::cout << avg_opt << std::endl;
    }
    else // used to communicate through stdout
    {
        std::cout << avg_opt << "\t";
    }
}


// print the optimal actions (influencer's xi opinion) sequence (used by Python script to communicate through stdout)
void print_seq(std::vector<std::map<int, std::tuple<int, double, double>>>& g, int state, bool c)
{
    int pts = g.size();                          // time horizon: N+1 = number of states
    int N = pts - 1;
    int prev_idx = std::get<0>(g[N].at(state));  // last: size-1

    std::vector<double> xi_opt(N,0);             // optimal sequence of actions has N values (1 less than the states)
    xi_opt.back() = std::get<1>(g[N].at(state)); // add the last action

    // in the graph I have N+1 elements (from state 0 till state N); loop over the (reverse) time index
    for (int i=N-1; i > 0; i--)
    {
        xi_opt[i-1] = std::get<1>(g[i].at(prev_idx));
        prev_idx = std::get<0>(g[i].at(prev_idx));
    }

    if (!c)
    {
        std::cout << std::endl << " **           The optimal sequence is:            ** " << std::endl;
        std::cout << xi_opt << std::endl;
    }
    else // used to communicate through stdout
    {
        std::cout << xi_opt << "\t";
    }
}



/******************************************************** MAIN ********************************************************/

int main(int argc, char** argv)
{

    /************************************************* DEFAULT VALUES *************************************************/

    int B = 100;             // number of discrete opinion intervals (bins)
    int N = 50;              // number of actions/posts for the influencer (time horizon)
    int U = 10000;           // number of regular users

    double alpha = 0.2;      // weight on the prejudice in the opinion update
    double beta  = 0.7;      // weight on the current opinion in the opinion update
    
    const double p1 = 1.0;   // 'amplitude' of psi: needs to be 1.0
    double p2 = 0.5;         // width of rect, can be specified by command line

    double d1_over_d2 = 0.5; // cardinality/weight ratio of the first delta distribution with respect to the toal

    double xi_target = 1.0;  // target influencer's opinion value, where she wants to attract the users

                             // prejudice values for the two deltas, it coincides also with the inititial opinion
    double z_d1 = -1.0;      // dummy init to allow for cmd line specification
    double z_d2 = -1.0;      // dummy init
    bool move_back = false;  // decide if users a user feels the attraction of the prejudice even if not reach by a post

    double x_d1 = -1.0;      // dummy init to allow for cmd line specification
    double x_d2 = -1.0;      // dummy init

    bool debug = false;     // for debug prints
    bool all_trans = false; // print all transitions (also those overwritten)
    bool keep_max  = true;  // rule to decide which of the multiple link to keep

    bool call = false;      // use stdout to pass the results when the program is being called (by Phyton script)


    /************************************************* PARSE ARGUMENTS ************************************************/
    
    for (int i = 1; i < argc; i++)
    {
        if ((strcmp(argv[i], "--help") == 0) || (strcmp(argv[i], "-h") == 0))
            script_info();

        else if (strcmp(argv[i], "--recall") == 0) // flag
            move_back = true;
        
        else if (strcmp(argv[i], "--debug") == 0) // flag
            debug = true;

        else if ((strcmp(argv[i], "--intervals") == 0) || (strcmp(argv[i], "-i") == 0))
        {
            B = int_arg(i, argc, argv);
            if (B < 1)
                err_and_exit("ERROR: the number of bins needs to be > 1");
        }
        else if ((strcmp(argv[i], "--n-iter") == 0) || (strcmp(argv[i], "-n") == 0))
        {
            N = int_arg(i, argc, argv);
            if (N < 1)
                err_and_exit("ERROR: the number of iterations needs to be > 1");
        }
        else if ((strcmp(argv[i], "--users") == 0) || (strcmp(argv[i], "-u") == 0))
            U = int_arg(i, argc, argv);

        else if ((strcmp(argv[i], "--alpha") == 0) || (strcmp(argv[i], "-a") == 0))
        {
            alpha = double_arg(i, argc, argv);
            in_01(alpha, "alpha");
        }
        else if ((strcmp(argv[i], "--beta") == 0) || (strcmp(argv[i], "-b") == 0))
        {
            beta = double_arg(i, argc, argv);
            in_01(beta, "beta");
        }
        else if ((strcmp(argv[i], "--ratio") == 0) || (strcmp(argv[i], "-r") == 0))
        {
            d1_over_d2 = double_arg(i, argc, argv);
            in_01(d1_over_d2, "cardinality ratio");
        }
        else if (strcmp(argv[i], "--xitarget") == 0)
        {
            xi_target = double_arg(i, argc, argv);
            in_01(xi_target, "xi target");
        }
        else if (strcmp(argv[i], "-z1") == 0)
        {
            z_d1 = double_arg(i, argc, argv);
            in_01(z_d1, "delta 1 prejudice");
        }
        else if (strcmp(argv[i], "-z2") == 0)
        {
            z_d2 = double_arg(i, argc, argv);
            in_01(z_d2, "delta 2 prejudice");
        }
        else if (strcmp(argv[i], "-x1") == 0)
        {
            x_d1 = double_arg(i, argc, argv);
            in_01(x_d1, "delta 1 init opinion");
        }
        else if (strcmp(argv[i], "-x2") == 0)
        {
            x_d2 = double_arg(i, argc, argv);
            in_01(x_d2, "delta 2 init opinion");
        }
        else if (strcmp(argv[i], "-p2") == 0)
        {
            p2 = double_arg(i, argc, argv);
            in_01(p2, "parameter b");
        }
        else if (strcmp(argv[i], "--call") == 0)
        {
            call = true; // flag
        }
        else { // positional argument or unknown options passed
            std::string msg = "FATAL ERROR: unknown argument passed: ";
            err_and_exit(msg += argv[i]);
        }
    }

    // check on \alpha and \beta which needs to be < 1
    double a_b = alpha + beta;
    if(a_b > 1.0)
    {
        std::string msg = "FATAL ERROR: alpha+beta > 1 (" + std::to_string(a_b) +")";
        err_and_exit(msg);
    }

    double dx = 1.0 / (double)B;
    if (z_d1 < 0) z_d1 = 1.0/(2.0*(double)B); // default: 1st middle bin value
    if (z_d2 < 0) z_d2 = 1.0 - z_d1;          // default: last middle bin value

    if (x_d1 < 0) x_d1 = z_d1;                // default: same as prejudice
    if (x_d2 < 0) x_d2 = z_d2; 


  /******************************************** TRANSITION PATTERN ****************************************************/

    // mid-point of each opinion interval, this is also the opinion of a group when in bin 'idx' (= possible actions xi)
    std::vector<double> mid_bin(B,0);
    for (unsigned i=0; i < mid_bin.size(); i++) { mid_bin[i] = (1.0+2.0*(double)i) / (2.0*(double)B);}


    // STATES of the system (possible combinations of d1 and d2 occupation of the B bins) = B*B states
    std::vector<std::pair<int, int>> states(B*B,std::make_pair(-1.0,-1.0)); // pair as (bin of d1, bin of d2)=idx state
    for (int i=0; i < B*B; i++) { states[i] = std::make_pair(i / B, i % B);}


    // TRANSITION STRUCTURE: dict[source] -> provides a vector with all the B*B possible source state
    // each element of the vector (idx representing the source state) is a map (dictionary) where 'keys' are the desti-
    // nation states and the 'values' are pairs containing the action 'xi' and 'reward' (\mathbb{E}x) for the transition
    std::vector<std::map<int, std::pair<double, double>>> trans(B*B, std::map<int, std::pair<double, double>>({})); 
    

    // for all 'i'(source states) find the destination 'j' states and fill up the transition matrix
    // IF there are ties: multiple actions leading to the same state (i.e., multiple links) we keep the transition with
    // the highest \Delta_s (anyways those trasitions are equivalent, it is arbitrary which one to keep).
    for (int i=0; i < B*B; i++) // 'source' states, which are in the format (bin d1, bin d2) in the 'states' vector
    {   

        for (int j=0; j < B; j++) // loop over x_i possible actions at each 'i' state
        {
            double xi = mid_bin[j]; // influencer's action

            double x_d1 = mid_bin[states[i].first];     // user opinion of the first delta (.first)
            double ds_d1 = ds(alpha, beta, z_d1, x_d1, xi, move_back, p1, p2, B);
            int d1_dest_idx = x_to_idx(x_d1+ds_d1, B); // index of the destination state in the middle-bin vector

            double x_d2 = mid_bin[states[i].second];    // user opinion of the first delta (.second)
            double ds_d2 = ds(alpha, beta, z_d2, x_d2, xi, move_back, p1, p2, B);
            int d2_dest_idx = x_to_idx(x_d2+ds_d2, B); // index of the destination state in the middle-bin vector

            // the state vector is a flattered vector, so we combine the previous two idices
            int state_to = d1_dest_idx * B + d2_dest_idx; 

            if (debug && all_trans && !call)
            {
                std::cout << "new x1: " << x_d1+ds_d1 << " idx1 " << d1_dest_idx << "\tnew x2: " << x_d2+ds_d2 
                            << " idx2 " << d2_dest_idx << std::endl;
                std::cout << "transition from " << i << " to " << state_to << std::endl << std::endl;
            }
            
            // 'reward' function: consider the mean opinion of the users
            double reward = d1_over_d2 * (x_d1+ds_d1) + (1.0-d1_over_d2) * (x_d2+ds_d2);
            
            if (trans[i].find(state_to) == trans[i].end()) // the key is not in the transition structure
            {
                trans[i][state_to] = std::make_pair(xi, reward); // add the key-value pair to the map
            }
            else // key 'state_to' is already present
            { 
                if (keep_max)
                {
                    if (reward > trans[i].at(state_to).second) // check if the reward is greater
                    {
                        trans[i].at(state_to) = std::make_pair(xi, reward);
                        if (debug && !call)
                            std::cout << "\tmultiple links actually occur" << std::endl;
                    }
                }
            }
        }
    }

    if (!call)
        std::cout << std::endl << " **    the transition matrix has been defined     ** " << std::endl << std::endl;

    if (debug && !call)
    {
        for (int i=0; i < B*B; i++) // loop over the src state
        { 
            // print the source state in terms of (position d1, position d2)
            std::cout << "state " << i << ": " << "(" << states[i].first << "," << states[i].second << ")" << " -> ";
            
            for (auto& [k, v] : trans[i]) // loop over the map
            {
                std::cout << k << "-(" << v.first << "," << v.second << ") "; // k is the reached state from i
            }
            std::cout << std::endl << std::endl;
        }
    }


    /********************************************* UNFOLD OVER TIME ***************************************************/

    // DATA STRUCTURE: the index of the vector are time instants, each conatining a map, with the REACHED state until
    // that instant (key) with information on (A) the state they come from, (B) action chosen, and (C) aggregated reward
    // as a std::tuple (the value in the key-value pair). For each state at n+1 we need to keep only one transition to
    // that state, (highest \Delta_s). So each element is a tuple (state_from, action, (highest) aggregated reward).
    // The 'action' in the structure is the action that led to that state
    std::vector<std::map<int, std::tuple<int, double, double>>> graph(N+1, std::map<int, std::tuple<int, double, double>>({}));

    // choose the initial state, according to the z_d1 and z_d2 values (--> upgrade to arbitrary init op)
    int init_state = x_to_idx(x_d1, B) * B + x_to_idx(x_d2, B);
    if (debug && !call)
    {
        std::cout << "z d1: " << z_d1 << "(" << x_to_idx(z_d1, B) << ")" << "\tz d2: " << z_d2 << "(" 
                    << x_to_idx(z_d2, B) << ")" << std::endl;

        std::cout << "x d1: " << x_d1 << "(" << x_to_idx(x_d1, B) << ")" << "\tz d2: " << x_d2 << "(" 
                    << x_to_idx(x_d2, B) << ")" << std::endl;

        std::cout << "init state: " << init_state << std::endl << std::endl;
    }
    
    // we put as 'state_from' the state itself -> indicate the stopping condition (negative values signal 'not in use')
    graph[0][init_state] = std::make_tuple(init_state, 0.0, 0.0); // add value to the map

    for (int i=0; i < N; i++) // loop over columns: all time instants, (I update the i+1 element)
    {
        if (!call)
        {
            std::string str = std::to_string(i);
            std::cout << "          iteration " << std::setw(5) << std::setfill('0') << str << " in progress          " 
                        << std::endl;
        }

        // loop over the maps at each time instants, vals are filled at each iteration, iter i fills values at i+1
        for (auto& [k_from, v_info] : graph[i])
        {
            // where can I go from this state? use the 'transition' structure (loop over the map with k=k_from)
            for (auto& [k_to, v_trans] : trans[k_from])
            {
                
                double path_reward = v_trans.second; // here I take the expected x value associated with the k_to state
                                                     // 'k_to' is the possible target state, to which a trans can happen
                                                     // 'v_trans.second' is the ACTUAL non distretized value of E_x
                
                if (graph[i+1].find(k_to) == graph[i+1].end()) // check if in the next instant the key is already used
                {
                    // not present in the next instant -> add it (add the state)
                    graph[i+1][k_to] = std::make_tuple(k_from, v_trans.first, path_reward);
                }
                else // the key already exist -> keep only the entry (basically path) leading to the highest reward
                {
                    if (path_reward > std::get<2>(graph[i+1].at(k_to)))
                    {
                        graph[i+1][k_to] = std::make_tuple(k_from, v_trans.first, path_reward);
                    }
                }
            }
        }
    }

    if (debug && !call)
    {
        for (int i=0; i < N; i++) // print the entire time-sequence of states and transitions
        {
            std::cout << "n=" << i << "\t";
            for (auto& [k, v] : graph[i])
            {
                std::cout << k << "-"; // the info of the state is embedded in the key of the map
                print_tuple(v);
                std::cout << " ";
            }
            std::cout << std::endl << std::endl;
        }
    }


    /********************************************* OPTIMAL SEQUENCE ***************************************************/

    // identify the final state of interest (which could also not be reached) evaluating the upper bounds on x
    double upper_x_d1 = upper_x(alpha, beta, z_d1, B);
    double upper_x_d2 = upper_x(alpha, beta, z_d2, B);

    int fin_state = x_to_idx(upper_x_d1, B) * B + x_to_idx(upper_x_d2, B);

    if (!call)
    {
        std::cout << std::endl << "WARNING: the final desired state has not been reached" << std::endl;
        std::cout << "    STATE " << fin_state << ": in which d1=" << mid_bin[states[fin_state].first] << " and d2=" 
                    << mid_bin[states[fin_state].second] << std::endl;
    }

    int max_r_key  = 0;      // id of the state (the key in the map) whose -(|x^T-E_x|) is maximum
    double max_val = -100.0;
    for (auto& [k, v] : graph[N])
    {
        // find the state with the highest expected opinion value (E_x)
        // double candiate_val = std::get<2>(v);
        double candiate_val = -(std::abs(xi_target - std::get<2>(v)));
        if (candiate_val > max_val) {
            max_val = candiate_val;
            max_r_key = k;
        }
    }

    double x_d1_max = mid_bin[states[max_r_key].first];
    double x_d2_max = mid_bin[states[max_r_key].second];

    if (!call)
    {
        std::cout << std::endl;
        std::cout << " ** the sequence with the highest reward leads to ** " << std::endl;
        std::cout << "    STATE " << max_r_key << ": in which d1=" << x_d1_max << " and d2="  << x_d2_max << std::endl;
        std::cout << "    the average final opinion is: " << d1_over_d2*x_d1_max+ (1.0-d1_over_d2)*x_d2_max << std::endl;
    }
    
    print_seq(graph, max_r_key, call);                                  // print optimal action (xi) sequence
    print_opt_avg(graph, max_r_key, states, mid_bin, d1_over_d2, call); // print average population opinion
    std::vector<int> opt_states = print_states(graph, max_r_key, call);

    if (xi_target == 1.0) // otherwise we can have oscillatory behavior, not necessarily the first hit is the optimal one
    {
        int first_hit = N+1;
        for (unsigned i=0; i < graph.size(); i++)
        {
            auto it = graph[i].find(max_r_key);
            if (it != graph[i].end())
            {
                first_hit = i;
                break; // we found the value of interest
            }
        }
        if (!call)
            std::cout << std::endl << " * optimal state hit at i=" << first_hit << " the first time * " << std::endl;

        // check that the first hit corresponds to that we get in the optimal solution
        if(opt_states[first_hit] != max_r_key)
            err_and_exit("FATAL ERROR: the optimal sequence is not the quickest possible " +
                        std::to_string(first_hit) + ", " + std::to_string(opt_states[first_hit]) + "vs" + std::to_string(max_r_key));

    }

    return 0;
}
