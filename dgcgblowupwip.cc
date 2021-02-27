#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>

using namespace dealii;

// The initial condition

template <int dim> class initialvalues: public Function<dim>
{
public: initialvalues () : Function<dim>() {};

virtual void value_list (const std::vector<Point<dim>> &points,std::vector<double> &values, const unsigned int component = 0) const;
};

template <int dim> void initialvalues<dim>::value_list (const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int) const
{
const unsigned int no_of_points = points.size();

    for (unsigned int point = 0; point < no_of_points; ++point)
	{
	const double x = points[point](0); const double y = points[point](1);

    //values[point] = 10*exp(-2*(x*x + y*y));
    values[point] = 10*(x*x + y*y)*exp(-0.5*(x*x + y*y));
	}
}

// The laplacian of the initial condition

template <int dim> class initialvalueslaplacian: public Function<dim>
{
public: initialvalueslaplacian () : Function<dim>() {};

virtual void value_list (const std::vector<Point<dim>> &points,std::vector<double> &values, const unsigned int component = 0) const;
};

template <int dim> void initialvalueslaplacian<dim>::value_list (const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int) const
{
const unsigned int no_of_points = points.size();

    for (unsigned int point = 0; point < no_of_points; ++point)
	{
	const double x = points[point](0); const double y = points[point](1);

    //values[point] = 80*(2*(x*x + y*y) - 1)*exp(-2*(x*x + y*y));
    values[point] = 10*(x*x*x*x + y*y*y*y + 2*x*x*y*y - 6*x*x - 6*y*y + 4)*exp(-0.5*(x*x + y*y));
	}
}

template <int dim> class dGcGblowup
{
public:
  	
    dGcGblowup ();
    void run ();

    // PDE co-efficients
    const double a = 1; // Diffusion coefficient

    // Discretisation parameters
    const unsigned int space_degree = 8; // Spatial polynomial degree
	const unsigned int time_degree = 1; // Temporal polynomial degree
    const unsigned int refine_every_n_timesteps = 2; // Potentially refine the mesh every n timesteps
    unsigned int timestep_number = 0; // The current timestep
    double time = 0; // The current time
    double dt = 0.125*0.1; // The current timestep length
	double dt_old = dt; // The timestep length on the last time interval

	// Error estimator thresholds
    double spatial_refinement_threshold = 0.00001; // The spatial refinement threshold
    double spatial_coarsening_threshold = 0.1*std::pow(2.0, -1.0*space_degree)*spatial_refinement_threshold; // The spatial coarsening threshold
	double temporal_refinement_threshold = 0.000001; // The temporal refinement threshold
	const double delta_residual_threshold = 1e-4; // The threshold for the delta equation residual above which we consider the delta equation as having no root

    // Nonlinear solver parameters
    const std::string nonlinear_solver = "picard"; // Choose whether the nonlinear solver uses the "picard", "newton" or "hybrid" method
    const unsigned int newton_every_x_steps = 4; // If using the hybrid method, does a Newton step every x iterations
    const unsigned int maximum_nonlinear_iterates = 35; // Maximum number of iterates the nonlinear solver will do before terminating
    const double nonlinear_residual_threshold = 1e-13; // The nonlinear solver will continue to iterate until the difference in solutions is less than ||U||*nonlinear_residual_threshold


    // ~~INTERNAL PARAMETERS~~ -- LEAVE ALONE!!


    // Error estimator parameters
	double est = 0; // The error estimator
	double space_est = 1e16; // The space estimator
	double time_est = 0; // The time estimator
	double r = 0; // The scaling parameter r
	double delta = 1.5; // The scaling parameter delta (the solution of the delta equation)
	double solution_integral = 0; // The (time) integral of the Linfty norm of the numerical solution
	double delta_residual = 0; // The residual arising from the numerical solution of the delta equation

    // Mesh change parameters
    bool mesh_change = true; // Parameter indicating if mesh change recently occured between triangulation_space and old_triangulation_space
    bool old_mesh_change = false; // Parameter indicating if mesh change recently occured between old_triangulation_space and old_old_triangulation_space

private:

    void setup_system_full (); // Initialises all vectors, distributes all degrees of freedom and computes the static part of the system matrix
	void setup_system_partial (); // Reinitialises vectors and redistributes degrees of freedom related to the current triangulation. Also recomputes the static part of the system matrix. Required if the mesh or time step length changes
	void create_static_system_matrix (); // Creates the static part of the system matrix, i.e., that which does not change between nonlinear iterations
    void create_temporal_mass_matrix (const FE_DGQ<1> &fe_time, const DoFHandler<1> &dof_handler_time, FullMatrix<double> &temporal_mass_matrix) const; // Computes the temporal mass matrix M_ij = (phi_i, phi_j) where {phi_i} is the standard basis for the temporal dG space
	void create_time_derivative_matrix (FullMatrix<double> &time_derivative_matrix) const; // Computes the "time derivative" matrix L_ij = (phi_i, d(phi_j)/dt) where {phi_i} is the standard basis for the temporal dG space
    void energy_project (const unsigned int &no_q_space_x, const Function<dim> &laplacian_function, Vector<double> &projection) const; // Computes the "energy projection" of the initial condition u0 to the finite element function U0 such that (grad(U_0), grad(V_0)) = (-laplacian(u_0), V_0) holds for all V_0
	void assemble_and_solve (const unsigned int &no_q_space_x, const unsigned int &no_q_time); // Assembles the right-hand side vector and solves the nonlinear system via iteration until the difference in solutions is below ||U||*nonlinear_residual_threshold
    void refine_initial_mesh (); // Refines the initial mesh and recomputes the energy projection of the initial condition until ||u_0 - U_0|| < spatial_coarsening_threshold
    void refine_mesh (); // Refines all cells with refinement_vector(cell_no) > spatial_refinement_threshold and coarsens all cells with refinement_vector(cell_no) < spatial_coarsening_threshold
    void prepare_for_next_time_step (); // Prepares the vectors, triangulations and dof_handlers for the next time step by setting them to previous values
	void output_solution () const; // Outputs the solution at final time on the current time step
    void reorder_solution_vector (const Vector<double> &spacetime_fe_function, BlockVector<double> &reordered_spacetime_fe_function, const DoFHandler<dim> &dof_handler_space, const DoFHandler<dim> &dof_handler, const FESystem<dim> &fe) const; // Helper function which reorders the spacetime FEM vector into a blockvector with each block representing a temporal node
	void extend_to_constant_in_time_function (Vector<double> &fe_function, Vector<double> &spacetime_fe_function) const; // Helper function which takes a spatial FEM function and expands it to a constant-in-time spacetime FEM function
	void compute_Q_values (const unsigned int &degree, const double &point, double &Q_value, double &Q_derivative_value, double &Q_second_derivative_value) const; // Compute the "Q" values and their various derivatives from the temporal reconstruction needed for the space and time estimators
	void compute_space_estimator (const unsigned int &no_q_space_x, const unsigned int &no_q_time, const bool &output_refinement_vector); // Computes the space estimator. Optional argument specifies whether we ouptut the refinement vector needed for spatial mesh refinement
	void compute_time_estimator (const unsigned int &no_q_space_x, const unsigned int &no_q_time); // Computes the time estimator
	void compute_estimator (); // Solves the delta equation to determine if the estimator can be computed and, if it can be, computes it and outputs it along with other values of interest

	Triangulation<dim> triangulation_space; Triangulation<dim> old_triangulation_space; Triangulation<dim> old_old_triangulation_space; // The current mesh, the mesh from the previous timestep and the mesh from the previous previous timestep
	Triangulation<1> triangulation_time; Triangulation<1> old_triangulation_time; // The current temporal mesh and the temporal mesh from the previous timestep

	DoFHandler<dim> dof_handler_space; DoFHandler<dim> old_dof_handler_space; DoFHandler<dim> old_old_dof_handler_space; 
	DoFHandler<1> dof_handler_time; DoFHandler<1> old_dof_handler_time; 
	DoFHandler<dim> dof_handler;

    FE_Q<dim> fe_space; FE_Q<dim> old_fe_space; FE_Q<dim> old_old_fe_space; 
	FE_DGQ<1> fe_time; FE_DGQ<1> old_fe_time; 
	FESystem<dim> fe;

	AffineConstraints<double> constraints;
	SparsityPattern sparsity_pattern;

	SparseMatrix<double> system_matrix; // The system matrix is subdivided into a static portion which does not change between newton iterations and a dynamic portion which does 
    SparseMatrix<double> static_system_matrix; // The static part of the system matrix

    SparseILU<double> preconditioner;

	BlockVector<double> solution; // The solution on the current timestep
    BlockVector<double> old_solution; // The solution on the previous timestep

	Vector<double> old_old_solution_plus; // The solution evaluated at final time on the previous previous timestep
    Vector<double> refinement_vector; // Vector used to refine the mesh
};

template <int dim> dGcGblowup<dim>::dGcGblowup ()
                :
				dof_handler_space (triangulation_space), old_dof_handler_space (old_triangulation_space), old_old_dof_handler_space (old_old_triangulation_space),
				dof_handler_time (triangulation_time), old_dof_handler_time (old_triangulation_time),
				dof_handler (triangulation_space),
				fe_space (space_degree), old_fe_space (space_degree), old_old_fe_space (space_degree),
				fe_time (time_degree), old_fe_time (time_degree),
				fe (fe_space, time_degree + 1)
{}

// Initialises all vectors, distributes all degrees of freedom and computes the static part of the system matrix

template <int dim> void dGcGblowup<dim>::setup_system_full ()
{
dof_handler_space.distribute_dofs (fe_space); old_dof_handler_space.distribute_dofs (old_fe_space); old_old_dof_handler_space.distribute_dofs (old_old_fe_space);
dof_handler_time.distribute_dofs (fe_time); old_dof_handler_time.distribute_dofs (old_fe_time);
dof_handler.distribute_dofs (fe); 

constraints.clear ();
DoFTools::make_hanging_node_constraints (dof_handler, constraints);
DoFTools::make_zero_boundary_constraints (dof_handler, constraints);
constraints.close ();

DynamicSparsityPattern dsp (dof_handler.n_dofs());
DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
sparsity_pattern.copy_from (dsp);

solution.reinit (time_degree + 1); old_solution.reinit (time_degree + 1);

    for (unsigned int r = 0; r < time_degree + 1; ++r)
    {
    solution.block(r).reinit (dof_handler_space.n_dofs()); old_solution.block(r).reinit (old_dof_handler_space.n_dofs());
    }

solution.collect_sizes (); old_solution.collect_sizes ();

old_old_solution_plus.reinit (old_old_dof_handler_space.n_dofs());
refinement_vector.reinit (triangulation_space.n_active_cells());

create_static_system_matrix ();
}

// Reinitialises vectors and redistributes degrees of freedom related to the current triangulation. Also recomputes the static part of the system matrix. Required if the mesh or time step length changes

template <int dim> void dGcGblowup<dim>::setup_system_partial ()
{
if (time_est > temporal_refinement_threshold) {dof_handler_time.distribute_dofs (fe_time);}

if (mesh_change == true)
{
dof_handler_space.distribute_dofs (fe_space); dof_handler.distribute_dofs (fe);

constraints.clear ();
DoFTools::make_hanging_node_constraints (dof_handler, constraints);
DoFTools::make_zero_boundary_constraints (dof_handler, constraints);
constraints.close ();

DynamicSparsityPattern dsp (dof_handler.n_dofs());
DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
sparsity_pattern.copy_from (dsp);

solution.reinit (time_degree + 1); 

    for (unsigned int r = 0; r < time_degree + 1; ++r)
    solution.block(r).reinit (dof_handler_space.n_dofs());

solution.collect_sizes ();

refinement_vector.reinit (triangulation_space.n_active_cells());
}

create_static_system_matrix ();
}

// Creates the static part of the system matrix, i.e., that which does not change between nonlinear iterations

template <int dim> void dGcGblowup<dim>::create_static_system_matrix ()
{
if (nonlinear_solver != "picard") {static_system_matrix.reinit (sparsity_pattern);}
if (nonlinear_solver == "picard") {system_matrix.reinit (sparsity_pattern);}

const QGauss<dim> quadrature_formula_space (space_degree + 1);

FEValues<dim> fe_values_space (fe_space, quadrature_formula_space, update_values | update_gradients | update_JxW_values);

const unsigned int no_q_space = quadrature_formula_space.size ();
const unsigned int dofs_per_cell = fe.dofs_per_cell;
const unsigned int dofs_per_cell_space = fe_space.dofs_per_cell;

FullMatrix<double> local_system_matrix (dofs_per_cell, dofs_per_cell);
FullMatrix<double> local_mass_matrix (dofs_per_cell_space, dofs_per_cell_space);
FullMatrix<double> local_laplace_matrix (dofs_per_cell_space, dofs_per_cell_space);
FullMatrix<double> temporal_mass_matrix (time_degree + 1, time_degree + 1);
FullMatrix<double> time_derivative_matrix (time_degree + 1, time_degree + 1);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

create_temporal_mass_matrix (fe_time, dof_handler_time, temporal_mass_matrix); 
if (time_degree > 0) {create_time_derivative_matrix (time_derivative_matrix);}

typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active (), final_space_cell = dof_handler_space.end ();
typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active ();

double cell_size = 0; double previous_cell_size = 0; double cell_size_check = 0;

    for (; space_cell != final_space_cell; ++cell, ++space_cell)
    {
	cell->get_dof_indices (local_dof_indices); cell_size = space_cell->measure ();
    cell_size_check = fabs(cell_size - previous_cell_size);

    if (cell_size_check > 1e-15)
    {
    local_system_matrix = 0; local_mass_matrix = 0; local_laplace_matrix = 0;

    fe_values_space.reinit (space_cell);

        for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
            for (unsigned int j = 0; j < i + 1; ++j)
            {
                for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
	            {
	            local_mass_matrix(i,j) += fe_values_space.shape_value(i,q_space)*fe_values_space.shape_value(j,q_space)*fe_values_space.JxW(q_space);
	            local_laplace_matrix(i,j) += a*fe_values_space.shape_grad(i,q_space)*fe_values_space.shape_grad(j,q_space)*fe_values_space.JxW(q_space);
	            }

            local_mass_matrix(j,i) = local_mass_matrix(i,j); local_laplace_matrix(j,i) = local_laplace_matrix(i,j);
            }

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
        const unsigned int comp_s_k = fe.system_to_component_index(k).second; const unsigned int comp_t_k = fe.system_to_component_index(k).first;

            for (unsigned int l = 0; l < dofs_per_cell; ++l)
            {
            const unsigned int comp_s_l = fe.system_to_component_index(l).second; const unsigned int comp_t_l = fe.system_to_component_index(l).first;

            local_system_matrix(k,l) += temporal_mass_matrix(comp_t_k,comp_t_l)*local_laplace_matrix(comp_s_k,comp_s_l);
            if (time_degree > 0) {local_system_matrix(k,l) += time_derivative_matrix(comp_t_k,comp_t_l)*local_mass_matrix(comp_s_k,comp_s_l);}
            if ((comp_t_k == 0) && (comp_t_l == 0)) {local_system_matrix(k,l) += local_mass_matrix(comp_s_k,comp_s_l);}
            }
        }
    }

    if (nonlinear_solver != "picard") {constraints.distribute_local_to_global (local_system_matrix, local_dof_indices, static_system_matrix);}
    if (nonlinear_solver == "picard") {constraints.distribute_local_to_global (local_system_matrix, local_dof_indices, system_matrix);}

    previous_cell_size = cell_size; 
    }

if (nonlinear_solver == "picard") {preconditioner.initialize (system_matrix);}
}

// Computes the temporal mass matrix M_ij = (phi_i, phi_j) where {phi_i} is the standard basis for the temporal dG space

template <int dim> void dGcGblowup<dim>::create_temporal_mass_matrix (const FE_DGQ<1> &fe_time, const DoFHandler<1> &dof_handler_time, FullMatrix<double> &temporal_mass_matrix) const
{
const QGauss<1> quadrature_formula_time (time_degree + 1);
FEValues<1> fe_values_time (fe_time, quadrature_formula_time, update_values | update_JxW_values);

typename DoFHandler<1>::active_cell_iterator time_cell = dof_handler_time.begin_active (); fe_values_time.reinit (time_cell);

    for (unsigned int r = 0; r < time_degree + 1; ++r)
        for (unsigned int s = 0; s < r + 1; ++s)
        {
            for (unsigned int q_time = 0; q_time < time_degree + 1; ++q_time)
	        temporal_mass_matrix(r,s) += fe_values_time.shape_value(r,q_time)*fe_values_time.shape_value(s,q_time)*fe_values_time.JxW(q_time);

	    temporal_mass_matrix(s,r) = temporal_mass_matrix(r,s);
        }
}

// Computes the "time derivative" matrix L_ij = (phi_i, d(phi_j)/dt) where {phi_i} is the standard basis for the temporal dG space

template <int dim> void dGcGblowup<dim>::create_time_derivative_matrix (FullMatrix<double> &time_derivative_matrix) const
{
const QGauss<1> quadrature_formula_time (time_degree + 1);
FEValues<1> fe_values_time (fe_time, quadrature_formula_time, update_values | update_gradients | update_JxW_values);

typename DoFHandler<1>::active_cell_iterator time_cell = dof_handler_time.begin_active (); fe_values_time.reinit (time_cell);

    for (unsigned int r = 0; r < time_degree + 1; ++r)
        for (unsigned int q_time = 0; q_time < time_degree + 1; ++q_time)
        {
        const double value = fe_values_time.shape_value(r,q_time)*fe_values_time.JxW(q_time);

            for (unsigned int s = 0; s < time_degree + 1; ++s)
            time_derivative_matrix(r,s) += value*fe_values_time.shape_grad(s,q_time)[0];
        }
}

// Computes the "energy projection" of the initial condition u0 to the finite element function U0 such that (grad(U_0), grad(V_0)) = (-laplacian(u_0), V_0) holds for all V_0

template <int dim> void dGcGblowup<dim>::energy_project (const unsigned int &no_q_space_x, const Function<dim> &laplacian_function, Vector<double> &projection) const
{
const QGauss<dim> quadrature_formula_space (no_q_space_x);

FEValues<dim> fe_values_space (fe_space, quadrature_formula_space, update_values | update_gradients | update_quadrature_points | update_JxW_values);

const unsigned int no_q_space = quadrature_formula_space.size ();
const unsigned int no_of_space_dofs = dof_handler_space.n_dofs ();
const unsigned int dofs_per_cell = fe_space.dofs_per_cell;

AffineConstraints<double> constraints; SparsityPattern sparsity_pattern;

constraints.clear ();
DoFTools::make_hanging_node_constraints (dof_handler_space, constraints);
DoFTools::make_zero_boundary_constraints (dof_handler_space, constraints);
constraints.close ();

DynamicSparsityPattern dsp (no_of_space_dofs);
DoFTools::make_sparsity_pattern (dof_handler_space, dsp, constraints, false);
sparsity_pattern.copy_from (dsp);

SparseMatrix<double> laplace_matrix; laplace_matrix.reinit (sparsity_pattern);
FullMatrix<double> local_laplace_matrix (dofs_per_cell, dofs_per_cell);
Vector<double> right_hand_side (no_of_space_dofs);
Vector<double> local_right_hand_side (dofs_per_cell);
std::vector<double> laplacian_values (no_q_space);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_space.begin_active (), final_cell = dof_handler_space.end ();

double cell_size = 0; double previous_cell_size = 0; double cell_size_check = 0;

    for (; cell != final_cell; ++cell)
    {
    fe_values_space.reinit (cell); cell->get_dof_indices (local_dof_indices); 
    cell_size = cell->measure (); cell_size_check = fabs(cell_size - previous_cell_size);

    laplacian_function.value_list (fe_values_space.get_quadrature_points(), laplacian_values);

    if (cell_size_check > 1e-15)
    {
    local_laplace_matrix = 0;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < i + 1; ++j)
            {
                for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
                local_laplace_matrix(i,j) += fe_values_space.shape_grad(i,q_space)*fe_values_space.shape_grad(j,q_space)*fe_values_space.JxW(q_space);

            local_laplace_matrix(j,i) = local_laplace_matrix(i,j);
            }
    }

        for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
        {
        const double value = laplacian_values[q_space]*fe_values_space.JxW(q_space);
        
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            local_right_hand_side(i) -= value*fe_values_space.shape_value(i,q_space);
        }

    constraints.distribute_local_to_global (local_laplace_matrix, local_right_hand_side, local_dof_indices, laplace_matrix, right_hand_side); local_right_hand_side = 0;

    previous_cell_size = cell_size;
    }

SolverControl solver_control (10000, 1e-20, false, false);
SolverBicgstab<> solver (solver_control);

SparseILU<double> preconditioner; preconditioner.initialize (laplace_matrix);
solver.solve (laplace_matrix, projection, right_hand_side, preconditioner);

constraints.distribute (projection);
}

// Assembles the right-hand side vector and solves the nonlinear system via iteration until the difference in solutions is below ||U||*nonlinear_residual_threshold

template <int dim> void dGcGblowup<dim>::assemble_and_solve (const unsigned int &no_q_space_x, const unsigned int &no_q_time)
{
if (nonlinear_solver == "newton") {deallog << "Calculating the numerical solution via Newton iteration..." << std::endl;}
if (nonlinear_solver == "picard") {deallog << "Calculating the numerical solution via Picard iteration..." << std::endl;}
if (nonlinear_solver == "hybrid") {deallog << "Calculating the numerical solution via hybrid Picard/Newton iteration..." << std::endl;}

const QGauss<dim> quadrature_formula_space (no_q_space_x); const QGauss<1> quadrature_formula_time (no_q_time);

FEValues<dim> fe_values_space (fe_space, quadrature_formula_space, update_values | update_quadrature_points | update_JxW_values);
FEValues<1> fe_values_time (fe_time, quadrature_formula_time, update_values | update_JxW_values);

const unsigned int no_of_dofs = dof_handler.n_dofs ();
const unsigned int no_q_space = quadrature_formula_space.size ();
const unsigned int dofs_per_cell = fe.dofs_per_cell;

FullMatrix<double> local_system_matrix (dofs_per_cell, dofs_per_cell);
Vector<double> temporary_solution (no_of_dofs);
Vector<double> right_hand_side (no_of_dofs);
Vector<double> static_right_hand_side (no_of_dofs);
Vector<double> residual_vector (no_of_dofs);
Vector<double> solution_values (no_q_space*no_q_time);
Vector<double> local_right_hand_side (dofs_per_cell);
std::vector<double> fe_values_spacetime (dofs_per_cell*no_q_space*no_q_time);
std::vector<double> nonlinearity_values (no_q_space*no_q_time);
std::vector<double> old_solution_plus_values (no_q_space);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

const Functions::FEFieldFunction<dim> old_solution_plus_function (old_dof_handler_space, old_solution.block(time_degree));

if (mesh_change == false) // Extend the numerical solution at final time on the previous interval to a constant-in-time function for use as an initial guess in the nonlinear iteration
{
switch (time_degree) {case 0: temporary_solution = old_solution.block(time_degree); break; default: extend_to_constant_in_time_function (old_solution.block(time_degree), temporary_solution);}
}
else // If the mesh has changed, we do as above but must first interpolate to the current finite element space
{
VectorTools::interpolate_to_different_mesh (old_dof_handler_space, old_solution.block(time_degree), dof_handler_space, solution.block(0));

switch (time_degree) {case 0: temporary_solution = solution.block(0); break; default: extend_to_constant_in_time_function (solution.block(0), temporary_solution);}
}

typename DoFHandler<1>::active_cell_iterator time_cell = dof_handler_time.begin_active (); fe_values_time.reinit (time_cell);

unsigned int iteration_number = 1; double residual = 0; const double max = old_solution.block(time_degree).linfty_norm();

    for (; iteration_number < maximum_nonlinear_iterates; ++iteration_number)
    {
    if (nonlinear_solver == "newton" || (nonlinear_solver == "hybrid" && iteration_number % newton_every_x_steps == 0) || (nonlinear_solver == "hybrid" && iteration_number % newton_every_x_steps == 1)) {system_matrix.reinit (sparsity_pattern); system_matrix.add (1, static_system_matrix);} // Set the system matrix to the static part of the system matrix if using Newton iteration

    typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active ();
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active (), final_cell = dof_handler.end ();

    residual_vector = temporary_solution;

    double cell_size = 0; double previous_cell_size = 0; double cell_size_check = 0; right_hand_side = 0;
    
        for (; cell != final_cell; ++cell, ++space_cell)
        {
        fe_values_space.reinit (space_cell);

        cell->get_dof_indices (local_dof_indices); cell_size = space_cell->measure ();
        cell_size_check = fabs(cell_size - previous_cell_size);
 
        if (iteration_number == 1) {old_solution_plus_function.value_list (fe_values_space.get_quadrature_points(), old_solution_plus_values);}

        if (cell_size_check > 1e-15)
        {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
            const unsigned int comp_s_k = fe.system_to_component_index(k).second; const unsigned int comp_t_k = fe.system_to_component_index(k).first;

                for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
                    for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
                    fe_values_spacetime[k + q_space*dofs_per_cell + q_time*dofs_per_cell*no_q_space] = fe_values_space.shape_value(comp_s_k,q_space)*fe_values_time.shape_value(comp_t_k,q_time);
            }
        }

            for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
                for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
                {
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    solution_values(q_space + q_time*no_q_space) += temporary_solution(local_dof_indices[k])*fe_values_spacetime[k + q_space*dofs_per_cell + q_time*dofs_per_cell*no_q_space];

                nonlinearity_values[q_space + q_time*no_q_space] = solution_values(q_space + q_time*no_q_space)*solution_values(q_space + q_time*no_q_space)*fe_values_space.JxW(q_space)*fe_values_time.JxW(q_time);
                if (nonlinear_solver == "newton" || (nonlinear_solver == "hybrid" && iteration_number % newton_every_x_steps == 0)) {solution_values(q_space + q_time*no_q_space) *= fe_values_space.JxW(q_space)*fe_values_time.JxW(q_time);}
                }

        // Assemble the local contributions of the dynamic part of the system matrix if using Newton iteration

        if (nonlinear_solver == "newton" || (nonlinear_solver == "hybrid" && iteration_number % newton_every_x_steps == 0))
        {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
                for (unsigned int l = 0; l < k + 1; ++l)
                {
                    for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
                        for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
                        local_system_matrix(k,l) += solution_values(q_space + q_time*no_q_space)*fe_values_spacetime[k + q_space*dofs_per_cell + q_time*dofs_per_cell*no_q_space]*fe_values_spacetime[l + q_space*dofs_per_cell + q_time*dofs_per_cell*no_q_space];

                local_system_matrix(l,k) = local_system_matrix(k,l);
                }

        // Distribute the local contributions of the dynamic part of the system matrix to the global system matrix if using Newton iteration
        local_system_matrix *= -2; constraints.distribute_local_to_global (local_system_matrix, local_dof_indices, system_matrix); local_system_matrix = 0; 
        }        
       
        // If on the first nonlinear iteration, assemble the local contributions of the static right-hand side vector and place them in the global static right-hand side vector

        if (iteration_number == 1)
        {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
            const unsigned int comp_t_k = fe.system_to_component_index(k).first;

            if (comp_t_k == 0)
            {
            const unsigned int comp_s_k = fe.system_to_component_index(k).second;

                for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
                local_right_hand_side(k) += old_solution_plus_values[q_space]*fe_values_space.shape_value(comp_s_k,q_space)*fe_values_space.JxW(q_space);
            }
            }

        constraints.distribute_local_to_global (local_right_hand_side, local_dof_indices, static_right_hand_side); local_right_hand_side = 0;
        } 

        // Assemble the local contributions of the dynamic part of the right-hand side vector

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
                for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
                    for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
                    local_right_hand_side(k) += nonlinearity_values[q_space + q_time*no_q_space]*fe_values_spacetime[k + q_space*dofs_per_cell + q_time*dofs_per_cell*no_q_space];
       
        if (nonlinear_solver == "newton" || (nonlinear_solver == "hybrid" && iteration_number % newton_every_x_steps == 0)) {local_right_hand_side *= -1;}

        // Distribute the local contributions of the dynamic parts of the right-hand side vector to the global right-hand side vector
        constraints.distribute_local_to_global (local_right_hand_side, local_dof_indices, right_hand_side); local_right_hand_side = 0; solution_values = 0;

        previous_cell_size = cell_size;
        } 

    right_hand_side.add (1, static_right_hand_side); // Add the static right-hand side vector to the right-hand side vector

    // Solve the matrix-vector system

    SolverControl solver_control (10000, 0.001*max*nonlinear_residual_threshold, false, false);
    SolverBicgstab<> solver (solver_control);

    if (nonlinear_solver == "newton" || (nonlinear_solver == "hybrid" && iteration_number % newton_every_x_steps == 0) || (nonlinear_solver == "hybrid" && iteration_number % newton_every_x_steps == 1)) {preconditioner.initialize (system_matrix);}

    solver.solve (system_matrix, temporary_solution, right_hand_side, preconditioner);

    constraints.distribute (temporary_solution);

    // Compute the residual and terminate the nonlinear iteration when the difference in solutions is sufficiently small
    residual_vector.add (-1, temporary_solution); residual = residual_vector.linfty_norm (); if (residual < max*nonlinear_residual_threshold) {break;}
    }

switch(time_degree) {case 0: solution.block(0) = temporary_solution; break; default: reorder_solution_vector (temporary_solution, solution, dof_handler_space, dof_handler, fe);}

if (iteration_number == maximum_nonlinear_iterates) {deallog << "...converged in the maximum number of allowed iterations (" << maximum_nonlinear_iterates << ") with a residual of " << residual << std::endl;} else {deallog << "...converged in " << iteration_number << " iterations with a residual of " << residual << std::endl;}
}

// Refines the initial mesh and recomputes the energy projection of the initial condition until ||u_0 - U_0|| < spatial_coarsening_threshold

template <int dim> void dGcGblowup<dim>::refine_initial_mesh ()
{
while (space_est > spatial_coarsening_threshold)
{
dof_handler_space.distribute_dofs (fe_space);

Vector<double> projection (dof_handler_space.n_dofs()); Vector<double> error (triangulation_space.n_active_cells());

deallog << std::endl << "Spatial Degrees of Freedom: " << dof_handler_space.n_dofs() << std::endl;
deallog << "Projecting the initial condition..." << std::endl;

energy_project (2*space_degree + 1, initialvalueslaplacian<dim>(), projection);

VectorTools::integrate_difference (dof_handler_space, projection, initialvalues<dim>(), error, QGauss<dim>(2*space_degree + 1), VectorTools::Linfty_norm); 
space_est = error.linfty_norm ();

deallog << "Initial Linfty Error: " << space_est << std::endl << std::endl;
if (space_est > spatial_coarsening_threshold) {deallog << "Initial Linfty error is too large. Refining the mesh..." << std::endl;} else {deallog << "Initial Linfty error is sufficiently small. Proceeding to the initial setup step." << std::endl;}

GridRefinement::refine (triangulation_space, error, spatial_coarsening_threshold);
triangulation_space.prepare_coarsening_and_refinement (); triangulation_space.execute_coarsening_and_refinement ();
}	
}

// Refines all cells with refinement_vector(cell_no) > spatial_refinement_threshold and coarsens all cells with refinement_vector(cell_no) < spatial_coarsening_threshold

template <int dim> void dGcGblowup<dim>::refine_mesh ()
{
if (timestep_number % refine_every_n_timesteps == 0)
{
GridRefinement::refine (triangulation_space, refinement_vector, spatial_refinement_threshold);
GridRefinement::coarsen (triangulation_space, refinement_vector, spatial_coarsening_threshold);

triangulation_space.prepare_coarsening_and_refinement (); triangulation_space.execute_coarsening_and_refinement ();
}

// Just because we TRY to refine the mesh DOES NOT MEAN IT CHANGES (EVEN IF SOME CELLS ARE FLAGGED)! The routine below checks whether or not the mesh has REALLY been modified
// If the mesh has been modified, we change mesh_change to true

if (triangulation_space.n_active_cells() != old_triangulation_space.n_active_cells()) 
{
mesh_change = true;
}
else
{
typename Triangulation<dim>::active_cell_iterator cell = triangulation_space.begin_active (), final_cell = triangulation_space.end ();
typename Triangulation<dim>::active_cell_iterator old_cell = old_triangulation_space.begin_active ();

    for (; cell != final_cell; ++cell, ++old_cell)
    {
        for (unsigned int vertex = 0; vertex < 4; ++vertex)
        {
        if ((cell->vertex(vertex) - old_cell->vertex(vertex))*(cell->vertex(vertex) - old_cell->vertex(vertex)) > 1e-15) {mesh_change = true; break;}
        }

    if (mesh_change == true) {break;}
    }
}

// For the first time step only, we also refine the other two meshes in order to keep all three meshes the same

if (timestep_number == 0)
{
GridRefinement::refine (old_triangulation_space, refinement_vector, spatial_refinement_threshold);
GridRefinement::coarsen (old_triangulation_space, refinement_vector, spatial_coarsening_threshold);

old_triangulation_space.prepare_coarsening_and_refinement (); old_triangulation_space.execute_coarsening_and_refinement ();

GridRefinement::refine (old_old_triangulation_space, refinement_vector, spatial_refinement_threshold);
GridRefinement::coarsen (old_old_triangulation_space, refinement_vector, spatial_coarsening_threshold);

old_old_triangulation_space.prepare_coarsening_and_refinement (); old_old_triangulation_space.execute_coarsening_and_refinement ();
}
}

// Prepares the vectors, triangulations and dof_handlers for the next time step by setting them to previous values

template <int dim> void dGcGblowup<dim>::prepare_for_next_time_step ()
{
// If the time step length has changed, set the old time step length to the current time step length and redistribute temporal dofs
if (dt != dt_old)
{
dt_old = dt; old_triangulation_time.clear (); old_triangulation_time.copy_triangulation (triangulation_time); old_dof_handler_time.distribute_dofs (old_fe_time);
}

// If the mesh changed between old_triangulation_space and old_old_triangulation_space set old_old_triangulation_space = old_triangulation_space and redistribute dofs. Either way, also reset the relevant vectors.
if (old_mesh_change == true)
{
old_old_triangulation_space.clear (); old_old_triangulation_space.copy_triangulation (old_triangulation_space);
old_old_dof_handler_space.distribute_dofs (old_old_fe_space);

old_old_solution_plus.reinit (old_old_dof_handler_space.n_dofs());
old_old_solution_plus = old_solution.block(time_degree);
}
else
{
old_old_solution_plus = old_solution.block(time_degree); 
}

// If the mesh changed between triangulation_space and old_triangulation_space set old_triangulation_space = triangulation_space and redistribute dofs. Either way, also reset the relevant vectors.
if (mesh_change == true)
{
old_triangulation_space.clear (); old_triangulation_space.copy_triangulation (triangulation_space);
old_dof_handler_space.distribute_dofs (old_fe_space);

old_solution.reinit (time_degree + 1);

    for (unsigned int r = 0; r < time_degree + 1; ++r)
    old_solution.block(r).reinit (old_dof_handler_space.n_dofs());

old_solution.collect_sizes ();

old_solution = solution;
}
else
{
old_solution = solution;
}

spatial_refinement_threshold *= r; spatial_coarsening_threshold *= r; temporal_refinement_threshold *= r; // Multiply all thresholds by the scaling parameter r_m.
old_mesh_change = mesh_change; mesh_change = false; // Reset the mesh change parameters in preparation for the next time step
}

// Outputs the solution at final time on the current time step

template <int dim> void dGcGblowup<dim>::output_solution () const
{
DataOut<dim> data_out; data_out.attach_dof_handler (dof_handler_space); data_out.add_data_vector (solution.block(time_degree), "u_h"); data_out.build_patches ();

const std::string filename = "solution-" + Utilities::int_to_string (timestep_number, 3) + ".gnuplot";

std::ofstream gnuplot_output (filename.c_str()); data_out.write_gnuplot (gnuplot_output);
}

// Helper function which reorders the spacetime FEM vector into a blockvector with each block representing a temporal node

template <int dim> void dGcGblowup<dim>::reorder_solution_vector (const Vector<double> &spacetime_fe_function, BlockVector<double> &reordered_spacetime_fe_function, const DoFHandler<dim> &dof_handler_space, const DoFHandler<dim> &dof_handler, const FESystem<dim> &fe) const
{
const unsigned int dofs_per_cell = fe.dofs_per_cell;

std::vector<types::global_dof_index> local_dof_indices_space (fe_space.dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active ();
typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active (), final_cell = dof_handler.end ();

    for (; cell != final_cell; ++cell, ++space_cell)
    {
    cell->get_dof_indices (local_dof_indices); space_cell->get_dof_indices (local_dof_indices_space);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
        const unsigned int comp_s_i = fe.system_to_component_index(i).second; const unsigned int comp_t_i = fe.system_to_component_index(i).first;

        reordered_spacetime_fe_function.block(comp_t_i)(local_dof_indices_space[comp_s_i]) = spacetime_fe_function(local_dof_indices[i]);
        }
    }
}

// Helper function which takes a spatial FEM function and expands it to a constant-in-time spacetime FEM function

template <int dim> void dGcGblowup<dim>::extend_to_constant_in_time_function (Vector<double> &fe_function, Vector<double> &spacetime_fe_function) const
{
const unsigned int dofs_per_cell = fe.dofs_per_cell;

std::vector<types::global_dof_index> local_dof_indices_space (fe_space.dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active ();
typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active (), final_cell = dof_handler.end ();

    for (; cell != final_cell; ++cell, ++space_cell)
    {
    cell->get_dof_indices (local_dof_indices); space_cell->get_dof_indices (local_dof_indices_space);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
        const unsigned int comp_s_i = fe.system_to_component_index(i).second;

        spacetime_fe_function(local_dof_indices[i]) = fe_function(local_dof_indices_space[comp_s_i]);
        }
    }
}

// Compute the "Q" values and their various derivatives from the temporal reconstruction needed for the space and time estimators

template <int dim> void dGcGblowup<dim>::compute_Q_values (const unsigned int &degree, const double &point, double &Q_value, double &Q_derivative_value, double &Q_second_derivative_value) const
{
switch(degree)
{
case 0: Q_value = point - 1.0; Q_derivative_value = 1.0; Q_second_derivative_value = 0; break;
case 1: Q_value = 1.5*point*point - point - 0.5; Q_derivative_value = 3.0*point - 1.0; Q_second_derivative_value = 3.0; break;
default: double value = 0; double old_value = point; double old_old_value = 1.0; double derivative_value = 0; double old_derivative_value = 1.0; double second_derivative_value = 0; double old_second_derivative_value = 0;

    for (unsigned int n = 2; n < degree + 2; ++n)
    {
    value = (2.0 - 1.0/n)*point*old_value - (1.0 - 1.0/n)*old_old_value;
    derivative_value = point*old_derivative_value + n*old_value;
    second_derivative_value = (n + 1)*old_derivative_value + point*old_second_derivative_value;
    old_old_value = old_value; old_value = value; old_derivative_value = derivative_value; old_second_derivative_value = second_derivative_value;

    if (n == degree) {Q_value = -value; Q_derivative_value = -derivative_value; Q_second_derivative_value = -second_derivative_value;} 
    if (n == degree + 1) {Q_value += value; Q_derivative_value += derivative_value; Q_second_derivative_value += second_derivative_value;}
    }
}

Q_value *= 0.5*std::pow(-1.0, degree); Q_derivative_value *= std::pow(-1.0, degree); Q_second_derivative_value *= 2.0*std::pow(-1.0, degree);
}

// Computes the space estimator. Optional argument specifies whether we ouptut the refinement vector needed for spatial mesh refinement
// The space estimator is split into two parts: a cell residual and an edge residual both of which must be computed
// For efficiency is decomposed into two possibilities
// If mesh_change == false and old_mesh_change == false, all meshes are the same and so we just compute on the current grid
// If either mesh_change == true or old_mesh_change == true, some meshes are different so we must form the UNION MESH then interpolate all vectors to it and work over this grid

template <int dim> void dGcGblowup<dim>::compute_space_estimator (const unsigned int &no_q_space_x, const unsigned int &no_q_time, const bool &output_refinement_vector)
{
const QGauss<dim> quadrature_formula_space (no_q_space_x); const QGauss<dim-1> quadrature_formula_space_face (no_q_space_x); const QGaussLobatto<1> quadrature_formula_time (no_q_time);

FEValues<1> fe_values_time (fe_time, quadrature_formula_time, update_values | update_gradients | update_hessians | update_quadrature_points | update_JxW_values);
FEValues<1> old_fe_values_time (old_fe_time, quadrature_formula_time, update_values | update_gradients | update_JxW_values);

const unsigned int no_q_space = quadrature_formula_space.size ();
const unsigned int dofs_per_cell_space = fe_space.dofs_per_cell;

FullMatrix<double> temporal_mass_matrix_inv (time_degree + 1, time_degree + 1);
FullMatrix<double> old_temporal_mass_matrix_inv (time_degree + 1, time_degree + 1);

Vector<double> L2_projection_rhs (time_degree + 1); Vector<double> L2_projection_f (time_degree + 1); std::vector<double> old_old_solution_plus_values (no_q_space);
std::vector<double> Q_values (no_q_time); std::vector<double> Q_derivative_values (no_q_time); std::vector<double> Q_second_derivative_values (no_q_time);
std::vector<Tensor<1,dim>> solution_face_gradient_values (no_q_space_x); std::vector<Tensor<1,dim>> solution_face_gradient_neighbor_values (no_q_space_x);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell_space);

create_temporal_mass_matrix (fe_time, dof_handler_time, temporal_mass_matrix_inv); temporal_mass_matrix_inv.gauss_jordan ();

if (dt == dt_old) {old_temporal_mass_matrix_inv = temporal_mass_matrix_inv;} else {create_temporal_mass_matrix (old_fe_time, old_dof_handler_time, old_temporal_mass_matrix_inv); old_temporal_mass_matrix_inv.gauss_jordan ();}
    
typename DoFHandler<1>::active_cell_iterator time_cell = dof_handler_time.begin_active (); typename DoFHandler<1>::active_cell_iterator old_time_cell = old_dof_handler_time.begin_active ();
fe_values_time.reinit (time_cell); old_fe_values_time.reinit (old_time_cell);

    for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
    compute_Q_values (time_degree, (2/dt)*fe_values_time.quadrature_point(q_time)(0) - 1, Q_values[q_time], Q_derivative_values[q_time], Q_second_derivative_values[q_time]);

space_est = 0; solution_integral = 0; if (output_refinement_vector == true) {refinement_vector = 0;}

if (mesh_change == false && old_mesh_change == false) // If mesh_change == false and old_mesh_change == false, all meshes are the same and so we just compute on the current grid
{
FEValues<dim> fe_values_space (fe_space, quadrature_formula_space, update_values | update_hessians | update_quadrature_points);
FEFaceValues<dim> fe_values_space_face (fe_space, quadrature_formula_space_face, update_gradients);
FEFaceValues<dim> fe_values_space_face_neighbor (fe_space, quadrature_formula_space_face, update_gradients | update_normal_vectors);
FESubfaceValues<dim> fe_values_space_subface (fe_space, quadrature_formula_space_face, update_gradients | update_normal_vectors);

const unsigned int no_of_space_dofs = dof_handler_space.n_dofs();
  
const double h_min = GridTools::minimal_cell_diameter (triangulation_space); const double ell_h = log(2 + 1/h_min);

    for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
    {
    double space_est_at_q_time = 0; double deriv_space_est_at_q_time = 0;  

    Vector<double> recon_sol_at_q_time (no_of_space_dofs); Vector<double> recon_deriv_at_q_time (no_of_space_dofs);

    recon_sol_at_q_time = solution.block(0); recon_sol_at_q_time.add(-1, old_solution.block(time_degree)); recon_deriv_at_q_time = recon_sol_at_q_time; 
    recon_sol_at_q_time *= Q_values[q_time]; recon_deriv_at_q_time *= (1/dt)*Q_derivative_values[q_time]; 

        for (unsigned int i = 0; i < no_of_space_dofs; ++i)
	        for (unsigned int r = 0; r < time_degree + 1; ++r)
            {
            recon_sol_at_q_time(i) += solution.block(r)(i)*fe_values_time.shape_value(r,q_time);
            if (time_degree > 0) {recon_deriv_at_q_time(i) += solution.block(r)(i)*fe_values_time.shape_grad(r,q_time)[0];}
	        }

    const double recon_sol_at_q_time_max = recon_sol_at_q_time.linfty_norm();

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_space.begin_active (), final_cell = dof_handler_space.end ();

        for (; cell != final_cell; ++cell)
        {
        fe_values_space.reinit (cell); cell->get_dof_indices (local_dof_indices);

        double space_est_cell = 0; double deriv_space_est_cell = 0; double space_est_int = 0; double deriv_space_est_int = 0;
        const unsigned int cell_no = cell->active_cell_index (); const double h = cell->diameter();
        const double C_cell = fmin(1/a, h*h*ell_h/a); const double C_edge = fmin(1, h*ell_h);

        if (timestep_number > 1) {fe_values_space.get_function_values (old_old_solution_plus, old_old_solution_plus_values);} else {initialvalueslaplacian<dim>().value_list (fe_values_space.get_quadrature_points(), old_old_solution_plus_values);}

            for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
            {
            double space_est_at_q_pt = 0; double deriv_space_est_at_q_pt = 0; double jump = 0;
            std::vector<double> solution_at_q_pt (no_q_time); std::vector<double> old_solution_at_q_pt (no_q_time);

                for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
                    for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
                        for (unsigned int r = 0; r < time_degree + 1; ++r)
                        {
                        solution_at_q_pt[q_time] += solution.block(r)(local_dof_indices[i])*fe_values_space.shape_value(i,q_space)*fe_values_time.shape_value(r,q_time);
                        old_solution_at_q_pt[q_time] += old_solution.block(r)(local_dof_indices[i])*fe_values_space.shape_value(i,q_space)*old_fe_values_time.shape_value(r,q_time);
                        }
                
                for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
                {
                if (space_degree > 1) {space_est_at_q_pt += a*recon_sol_at_q_time(local_dof_indices[i])*trace(fe_values_space.shape_hessian(i,q_space));
                deriv_space_est_at_q_pt += a*recon_deriv_at_q_time(local_dof_indices[i])*trace(fe_values_space.shape_hessian(i,q_space));}

                space_est_at_q_pt -= recon_deriv_at_q_time(local_dof_indices[i])*fe_values_space.shape_value(i,q_space);

                if (time_degree > 1)
                {
                    for (unsigned int r = 0; r < time_degree + 1; ++r)
                    deriv_space_est_at_q_pt -= solution.block(r)(local_dof_indices[i])*fe_values_space.shape_value(i,q_space)*fe_values_time.shape_hessian(r,q_time)[0][0];
                }              
                }

            if (time_degree > 0) {deriv_space_est_at_q_pt -= (1/dt)*(1/dt)*Q_second_derivative_values[q_time]*(solution_at_q_pt[0] - old_solution_at_q_pt[no_q_time - 1]);}

            L2_projection_rhs = 0;

                for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
                {
                const double nonlinearity_value = solution_at_q_pt[q_time]*solution_at_q_pt[q_time]*fe_values_time.JxW(q_time);
                
                    for (unsigned int r = 0; r < time_degree + 1; ++r)
                    L2_projection_rhs(r) += nonlinearity_value*fe_values_time.shape_value(r,q_time);
                }

            temporal_mass_matrix_inv.vmult (L2_projection_f, L2_projection_rhs);

            jump = L2_projection_f(0) - (1/dt)*Q_derivative_values[0]*(solution_at_q_pt[0] - old_solution_at_q_pt[no_q_time - 1]);

                for (unsigned int r = 0; r < time_degree + 1; ++r)
                {
                space_est_at_q_pt += L2_projection_f(r)*fe_values_time.shape_value(r,q_time);

                if (time_degree > 0)
                {
                deriv_space_est_at_q_pt += L2_projection_f(r)*fe_values_time.shape_grad(r,q_time)[0];

                    for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
                    jump -= solution.block(r)(local_dof_indices[i])*fe_values_space.shape_value(i,q_space)*fe_values_time.shape_grad(r,0)[0];
                }
                }

            if (timestep_number > 1)
            {
            L2_projection_rhs = 0;

                for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
                {
                const double nonlinearity_value = old_solution_at_q_pt[q_time]*old_solution_at_q_pt[q_time]*old_fe_values_time.JxW(q_time);

                    for (unsigned int r = 0; r < time_degree + 1; ++r)
                    L2_projection_rhs(r) += nonlinearity_value*old_fe_values_time.shape_value(r,q_time);
                }

                for (unsigned int r = 0; r < time_degree + 1; ++r)
                {
                jump -= old_temporal_mass_matrix_inv(time_degree, r)*L2_projection_rhs(r);

                if (time_degree > 0)
                {
                    for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
                    jump += old_solution.block(r)(local_dof_indices[i])*fe_values_space.shape_value(i,q_space)*old_fe_values_time.shape_grad(r,no_q_time - 1)[0];
                }
                }

            jump += (1/dt_old)*Q_derivative_values[no_q_time - 1]*(old_solution_at_q_pt[0] - old_old_solution_plus_values[q_space]);
            }
            else
            {
            jump += a*old_old_solution_plus_values[q_space];
            } 

            space_est_at_q_pt += Q_values[q_time]*jump; deriv_space_est_at_q_pt += (1/dt)*Q_derivative_values[q_time]*jump;

            space_est_int = fmax(space_est_int, fabs(space_est_at_q_pt)); deriv_space_est_int = fmax(deriv_space_est_int, fabs(deriv_space_est_at_q_pt));
            }
     
        double space_est_face = 0; double deriv_space_est_face = 0;

            for (unsigned int face = 0; face < 4; ++face)
            {
            if (cell->face(face)->at_boundary() == false && cell->face(face)->has_children() == false && cell->neighbor_is_coarser(face) == false) // Both faces are the same size
            {
		    typename DoFHandler<dim>::active_cell_iterator cell_neighbor = cell->neighbor (face);
		    const unsigned int neighbor_face_no = cell->neighbor_face_no (face);
         
	        fe_values_space_face.reinit (cell, face); fe_values_space_face_neighbor.reinit (cell_neighbor, neighbor_face_no);

		    fe_values_space_face.get_function_gradients (recon_sol_at_q_time, solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (recon_sol_at_q_time, solution_face_gradient_neighbor_values);

                for (unsigned int q_space = 0; q_space < no_q_space_x; ++q_space)
                space_est_face = fmax(space_est_face, fabs(solution_face_gradient_values[q_space]*fe_values_space_face_neighbor.normal_vector(q_space) - solution_face_gradient_neighbor_values[q_space]*fe_values_space_face_neighbor.normal_vector(q_space)));

            fe_values_space_face.get_function_gradients (recon_deriv_at_q_time, solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (recon_deriv_at_q_time, solution_face_gradient_neighbor_values);

                for (unsigned int q_space = 0; q_space < no_q_space_x; ++q_space)
                deriv_space_est_face = fmax(deriv_space_est_face, fabs(solution_face_gradient_values[q_space]*fe_values_space_face_neighbor.normal_vector(q_space) - solution_face_gradient_neighbor_values[q_space]*fe_values_space_face_neighbor.normal_vector(q_space)));
            }
            if (cell->face(face)->at_boundary() == false && cell->face(face)->has_children() == false && cell->neighbor_is_coarser(face) == true) // The neighbor face is coarser than the current face
            {
            typename DoFHandler<dim>::active_cell_iterator cell_neighbor = cell->neighbor (face);
            std::pair<unsigned int, unsigned int> neighbor_face_no = cell->neighbor_of_coarser_neighbor (face);

	        fe_values_space_face.reinit (cell, face); fe_values_space_subface.reinit (cell_neighbor, neighbor_face_no.first, neighbor_face_no.second);

		    fe_values_space_face.get_function_gradients (recon_sol_at_q_time, solution_face_gradient_values); fe_values_space_subface.get_function_gradients (recon_sol_at_q_time, solution_face_gradient_neighbor_values);

                for (unsigned int q_space = 0; q_space < no_q_space_x; ++q_space)
                space_est_face = fmax(space_est_face, fabs(solution_face_gradient_values[q_space]*fe_values_space_subface.normal_vector(q_space) - solution_face_gradient_neighbor_values[q_space]*fe_values_space_subface.normal_vector(q_space)));

            fe_values_space_face.get_function_gradients (recon_deriv_at_q_time, solution_face_gradient_values); fe_values_space_subface.get_function_gradients (recon_deriv_at_q_time, solution_face_gradient_neighbor_values);

                for (unsigned int q_space = 0; q_space < no_q_space_x; ++q_space)
                deriv_space_est_face = fmax(deriv_space_est_face, fabs(solution_face_gradient_values[q_space]*fe_values_space_subface.normal_vector(q_space) - solution_face_gradient_neighbor_values[q_space]*fe_values_space_subface.normal_vector(q_space)));
            }
            if (cell->face(face)->at_boundary() == false && cell->face(face)->has_children() == true && cell->neighbor_is_coarser(face) == false) // The neighbor face is more refined than the current face 
            {
            const unsigned int no_of_subfaces = cell->face(face)->n_children();
            const unsigned int neighbor_face_no = cell->neighbor_of_neighbor (face);

                for (unsigned int subface = 0; subface < no_of_subfaces; ++subface)
                {
                typename DoFHandler<dim>::active_cell_iterator cell_neighbor = cell->neighbor_child_on_subface (face, subface);

                fe_values_space_subface.reinit (cell, face, subface); fe_values_space_face_neighbor.reinit (cell_neighbor, neighbor_face_no);

                fe_values_space_subface.get_function_gradients (recon_sol_at_q_time, solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (recon_sol_at_q_time, solution_face_gradient_neighbor_values);

                    for (unsigned int q_space = 0; q_space < no_q_space_x; ++q_space)
                    space_est_face = fmax(space_est_face, fabs(solution_face_gradient_values[q_space]*fe_values_space_subface.normal_vector(q_space) - solution_face_gradient_neighbor_values[q_space]*fe_values_space_subface.normal_vector(q_space)));

                fe_values_space_subface.get_function_gradients (recon_deriv_at_q_time, solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (recon_deriv_at_q_time, solution_face_gradient_neighbor_values);

                    for (unsigned int q_space = 0; q_space < no_q_space_x; ++q_space)
                    deriv_space_est_face = fmax(deriv_space_est_face, fabs(solution_face_gradient_values[q_space]*fe_values_space_subface.normal_vector(q_space) - solution_face_gradient_neighbor_values[q_space]*fe_values_space_subface.normal_vector(q_space)));
                }	
            }
            }
        
        space_est_cell = C_cell*space_est_int + C_edge*space_est_face; deriv_space_est_cell = C_cell*deriv_space_est_int + C_edge*deriv_space_est_face;

        if (output_refinement_vector == true) {refinement_vector(cell_no) += (space_est_cell*(2*recon_sol_at_q_time_max + space_est_cell) + deriv_space_est_cell)*fe_values_time.JxW(q_time);}

        space_est_at_q_time = fmax(space_est_at_q_time, space_est_cell); deriv_space_est_at_q_time = fmax(deriv_space_est_at_q_time, deriv_space_est_cell);
        }

    space_est += (space_est_at_q_time*(2*recon_sol_at_q_time_max + space_est_at_q_time) + deriv_space_est_at_q_time)*fe_values_time.JxW(q_time);
    solution_integral += recon_sol_at_q_time_max*fe_values_time.JxW(q_time);
    }

if (output_refinement_vector == true) {refinement_vector *= 1/dt;}
}
else // If either mesh_change == true or old_mesh_change == true, some meshes are different so we must form the UNION MESH then interpolate all vectors to it and work over this grid
{
// Form the union mesh

Triangulation<dim> union_triangulation;

if (mesh_change == true && old_mesh_change == false) {GridGenerator::create_union_triangulation (triangulation_space, old_triangulation_space, union_triangulation);}
if (mesh_change == false && old_mesh_change == true) {GridGenerator::create_union_triangulation (triangulation_space, old_old_triangulation_space, union_triangulation);}
if (mesh_change == true && old_mesh_change == true)
{
Triangulation<dim> intermediate_triangulation; 

GridGenerator::create_union_triangulation (old_old_triangulation_space, old_triangulation_space, intermediate_triangulation);
GridGenerator::create_union_triangulation (triangulation_space, intermediate_triangulation, union_triangulation);
}

// Create relevant union mesh dof_handler and fe objects

DoFHandler<dim> dof_handler_space_union (union_triangulation); FE_Q<dim> fe_space_union (space_degree); dof_handler_space_union.distribute_dofs (fe_space_union);

FEValues<dim> fe_values_space_union (fe_space_union, quadrature_formula_space, update_values | update_hessians | update_quadrature_points);
FEFaceValues<dim> fe_values_space_union_face (fe_space_union, quadrature_formula_space_face, update_gradients);
FEFaceValues<dim> fe_values_space_union_face_neighbor (fe_space_union, quadrature_formula_space_face, update_gradients | update_normal_vectors);
FESubfaceValues<dim> fe_values_space_union_subface (fe_space_union, quadrature_formula_space_face, update_gradients | update_normal_vectors);

const unsigned int no_of_union_space_dofs = dof_handler_space_union.n_dofs ();

BlockVector<double> solution_union (time_degree + 1); BlockVector<double> old_solution_union (time_degree + 1);

    for (unsigned int r = 0; r < time_degree + 1; ++r)
    {
    solution_union.block(r).reinit (no_of_union_space_dofs); old_solution_union.block(r).reinit (no_of_union_space_dofs);
    }

solution_union.collect_sizes (); old_solution_union.collect_sizes ();

Vector<double> old_old_solution_plus_union (no_of_union_space_dofs);
Vector<double> refinement_union_vector; if (output_refinement_vector == true) {refinement_union_vector.reinit (union_triangulation.n_active_cells());}

// Interpolate all current vectors to the union mesh

AffineConstraints<double> spatial_union_constraints;

spatial_union_constraints.clear ();
DoFTools::make_hanging_node_constraints (dof_handler_space_union, spatial_union_constraints);
DoFTools::make_zero_boundary_constraints (dof_handler_space_union, spatial_union_constraints);
spatial_union_constraints.close ();

    for (unsigned int r = 0; r < time_degree + 1; ++r)
    {
    VectorTools::interpolate_to_different_mesh (old_dof_handler_space, old_solution.block(r), dof_handler_space_union, spatial_union_constraints, old_solution_union.block(r));
    VectorTools::interpolate_to_different_mesh (dof_handler_space, solution.block(r), dof_handler_space_union, spatial_union_constraints, solution_union.block(r));
    }

VectorTools::interpolate_to_different_mesh (old_old_dof_handler_space, old_old_solution_plus, dof_handler_space_union, spatial_union_constraints, old_old_solution_plus_union);
}
}

// Computes the time estimator
// For efficiency is decomposed into two possibilities
// If mesh_change == false and old_mesh_change == false, all meshes are the same and so we just compute on the current grid
// If either mesh_change == true or old_mesh_change == true, some meshes are different so we must form the UNION MESH then interpolate all vectors to it and work over this grid

template <int dim> void dGcGblowup<dim>::compute_time_estimator (const unsigned int &no_q_space_x, const unsigned int &no_q_time)
{
const QGauss<dim> quadrature_formula_space (no_q_space_x); const QGaussLobatto<1> quadrature_formula_time (no_q_time);

FEValues<1> fe_values_time (fe_time, quadrature_formula_time, update_values | update_gradients | update_quadrature_points | update_JxW_values);
FEValues<1> old_fe_values_time (old_fe_time, quadrature_formula_time, update_values | update_gradients | update_JxW_values);

const unsigned int no_q_space = quadrature_formula_space.size();
const unsigned int dofs_per_cell_space = fe_space.dofs_per_cell;

FullMatrix<double> temporal_mass_matrix_inv (time_degree + 1, time_degree + 1);
FullMatrix<double> old_temporal_mass_matrix_inv (time_degree + 1, time_degree + 1);

Vector<double> estimator_values (no_q_time); // Holds the Linfty norm of the temporal residual at each temporal quadrature point
Vector<double> solution_values (no_q_space*no_q_time);
Vector<double> old_solution_values (no_q_space*no_q_time);
Vector<double> L2_projection_rhs (time_degree + 1);
Vector<double> L2_projection_f (time_degree + 1);
std::vector<double> L2_projection_f_values (no_q_time);
std::vector<double> old_old_solution_plus_values (no_q_space);
std::vector<double> Q_values (no_q_time);
std::vector<double> Q_derivative_values (no_q_time);
std::vector<types::global_dof_index> local_dof_indices_space (dofs_per_cell_space);

if (time_degree > 0)
{
create_temporal_mass_matrix (fe_time, dof_handler_time, temporal_mass_matrix_inv); temporal_mass_matrix_inv.gauss_jordan ();

if (dt == dt_old) {old_temporal_mass_matrix_inv = temporal_mass_matrix_inv;}
else {create_temporal_mass_matrix (old_fe_time, old_dof_handler_time, old_temporal_mass_matrix_inv); old_temporal_mass_matrix_inv.gauss_jordan ();}
}

typename DoFHandler<1>::active_cell_iterator time_cell = dof_handler_time.begin_active(); typename DoFHandler<1>::active_cell_iterator old_time_cell = old_dof_handler_time.begin_active();
fe_values_time.reinit (time_cell); old_fe_values_time.reinit (old_time_cell);

    for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
    compute_Q_values (time_degree, (2/dt)*fe_values_time.quadrature_point(q_time)(0) - 1, Q_values[q_time], Q_derivative_values[q_time], time_est);

if (mesh_change == false && old_mesh_change == false) // If mesh_change == false and old_mesh_change == false, all meshes are the same and so we just compute on the current grid
{
FEValues<dim> fe_values_space (fe_space, quadrature_formula_space, update_values | update_quadrature_points);

typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active (), final_space_cell = dof_handler_space.end ();

time_est = 0; double discrete_laplacian_jump_value = 0; double jump_value = 0; double nonlinearity_value = 0; double solution_time_derivative_value = 0; 

    // Loop over all cells, compute the Linfty norm of the temporal residual of the solution at each temporal quadrature point and store it in estimator_values

    for (; space_cell != final_space_cell; ++space_cell)
    {
    fe_values_space.reinit (space_cell);

    space_cell->get_dof_indices (local_dof_indices_space);

    old_solution_values = 0; solution_values = 0;

        for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
   	            for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
                    for (unsigned int r = 0; r < time_degree + 1; ++r)
                    {
                    if (timestep_number > 1) {old_solution_values(q_space + q_time*no_q_space) += old_solution.block(r)(local_dof_indices_space[i])*fe_values_space.shape_value(i,q_space)*old_fe_values_time.shape_value(r,q_time);}
                    solution_values(q_space + q_time*no_q_space) += solution.block(r)(local_dof_indices_space[i])*fe_values_space.shape_value(i,q_space)*fe_values_time.shape_value(r,q_time);
                    }

    if (timestep_number > 1) {fe_values_space.get_function_values (old_old_solution_plus, old_old_solution_plus_values);}
    else {fe_values_space.get_function_values (old_solution.block(time_degree), old_old_solution_plus_values); for (unsigned int q_space = 0; q_space < no_q_space; ++q_space) {old_solution_values (q_space + (no_q_time - 1)*no_q_space) = old_old_solution_plus_values[q_space];} initialvalueslaplacian<dim>().value_list (fe_values_space.get_quadrature_points(), old_old_solution_plus_values);}

        for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
        {
        if (timestep_number > 1)
        {
        switch (time_degree) {case 0: L2_projection_f(time_degree) = old_solution_values(q_space)*old_solution_values(q_space); break;
        default: L2_projection_f(time_degree) = 0; L2_projection_rhs = 0; solution_time_derivative_value = 0;

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            nonlinearity_value = old_solution_values(q_space + q_time*no_q_space)*old_solution_values(q_space + q_time*no_q_space)*old_fe_values_time.JxW(q_time);

	            for (unsigned int i = 0; i < time_degree + 1; ++i)
	            L2_projection_rhs(i) += nonlinearity_value*old_fe_values_time.shape_value(i, q_time);
            }

            for (unsigned int i = 0; i < time_degree + 1; ++i)
            L2_projection_f(time_degree) += old_temporal_mass_matrix_inv(time_degree, i)*L2_projection_rhs(i);

	        for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
                for (unsigned int r = 0; r < time_degree + 1; ++r)
                solution_time_derivative_value += old_solution.block(r)(local_dof_indices_space[i])*fe_values_space.shape_value(i,q_space)*old_fe_values_time.shape_grad(r,no_q_time - 1)[0];
        }

        discrete_laplacian_jump_value = -L2_projection_f(time_degree) + solution_time_derivative_value + (1/dt_old)*Q_derivative_values[no_q_time-1]*(old_solution_values(q_space) - old_old_solution_plus_values[q_space]);
        }
        else
        {
        discrete_laplacian_jump_value = a*old_old_solution_plus_values[q_space];
        }

        switch (time_degree) {case 0: for (unsigned int q_time = 0; q_time < no_q_time; ++q_time) {L2_projection_f_values[q_time] = solution_values(q_space)*solution_values(q_space);} break;
        default: L2_projection_rhs = 0; solution_time_derivative_value = 0;

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            nonlinearity_value = solution_values(q_space + q_time*no_q_space)*solution_values(q_space + q_time*no_q_space)*fe_values_time.JxW(q_time);

                for (unsigned int i = 0; i < time_degree + 1; ++i)
                L2_projection_rhs(i) += nonlinearity_value*fe_values_time.shape_value(i, q_time);
            }

        temporal_mass_matrix_inv.vmult (L2_projection_f, L2_projection_rhs);
        fe_values_time.get_function_values (L2_projection_f, L2_projection_f_values);

            for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
                for (unsigned int r = 0; r < time_degree + 1; ++r)
                solution_time_derivative_value += solution.block(r)(local_dof_indices_space[i])*fe_values_space.shape_value(i,q_space)*fe_values_time.shape_grad(r,0)[0];
        }

        jump_value = solution_values(q_space) - old_solution_values(q_space + (no_q_time - 1)*no_q_space);
        discrete_laplacian_jump_value += L2_projection_f_values[0] - solution_time_derivative_value - (1/dt)*Q_derivative_values[0]*jump_value;

	        for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            double f_reconstructed = solution_values(q_space + q_time*no_q_space) + Q_values[q_time]*jump_value; f_reconstructed *= f_reconstructed;
            estimator_values(q_time) = fmax(estimator_values(q_time), fabs(f_reconstructed - L2_projection_f_values[q_time] - Q_values[q_time]*discrete_laplacian_jump_value));
            }
        }
    }
}
else // If either mesh_change == true or old_mesh_change == true, some meshes are different so we must form the UNION MESH then interpolate all vectors to it and work over this grid
{
// Form the union mesh

Triangulation<dim> union_triangulation;

if (mesh_change == true && old_mesh_change == false) {GridGenerator::create_union_triangulation (triangulation_space, old_triangulation_space, union_triangulation);}
if (mesh_change == false && old_mesh_change == true) {GridGenerator::create_union_triangulation (triangulation_space, old_old_triangulation_space, union_triangulation);}
if (mesh_change == true && old_mesh_change == true)
{
Triangulation<dim> intermediate_triangulation;

GridGenerator::create_union_triangulation (old_old_triangulation_space, old_triangulation_space, intermediate_triangulation);
GridGenerator::create_union_triangulation (triangulation_space, intermediate_triangulation, union_triangulation);
}

// Create relevant union mesh dof_handler and fe objects

DoFHandler<dim> dof_handler_space_union (union_triangulation); FE_Q<dim> fe_space_union (space_degree); dof_handler_space_union.distribute_dofs (fe_space_union);

FEValues<dim> fe_values_space_union (fe_space_union, quadrature_formula_space, update_values | update_quadrature_points);

const unsigned int no_of_union_space_dofs = dof_handler_space_union.n_dofs ();

BlockVector<double> solution_union (time_degree + 1); BlockVector<double> old_solution_union (time_degree + 1);

    for (unsigned int r = 0; r < time_degree + 1; ++r)
    {
    solution_union.block(r).reinit (no_of_union_space_dofs); old_solution_union.block(r).reinit (no_of_union_space_dofs);
    }

solution_union.collect_sizes (); old_solution_union.collect_sizes ();

Vector<double> old_old_solution_plus_union (no_of_union_space_dofs);

AffineConstraints<double> spatial_union_constraints;

spatial_union_constraints.clear ();
DoFTools::make_hanging_node_constraints (dof_handler_space_union, spatial_union_constraints);
DoFTools::make_zero_boundary_constraints (dof_handler_space_union, spatial_union_constraints);
spatial_union_constraints.close ();

// Interpolate all current vectors to the union mesh

VectorTools::interpolate_to_different_mesh (old_old_dof_handler_space, old_old_solution_plus, dof_handler_space_union, spatial_union_constraints, old_old_solution_plus_union);

    for (unsigned int r = 0; r < time_degree + 1; ++r)
    {
    VectorTools::interpolate_to_different_mesh (old_dof_handler_space, old_solution.block(r), dof_handler_space_union, spatial_union_constraints, old_solution_union.block(r));
    VectorTools::interpolate_to_different_mesh (dof_handler_space, solution.block(r), dof_handler_space_union, spatial_union_constraints, solution_union.block(r));
    }

typename DoFHandler<dim>::active_cell_iterator union_space_cell = dof_handler_space_union.begin_active (), final_union_space_cell = dof_handler_space_union.end ();

time_est = 0; double discrete_laplacian_jump_value = 0; double jump_value = 0; double nonlinearity_value = 0; double solution_time_derivative_value = 0;

    // Loop over all cells, compute the Linfty norm of the temporal residual of the solution at each temporal quadrature point and store it in estimator_values

    for (; union_space_cell != final_union_space_cell; ++union_space_cell)
    {
    fe_values_space_union.reinit (union_space_cell);

    union_space_cell->get_dof_indices (local_dof_indices_space);

    old_solution_values = 0; solution_values = 0;

        for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
   	            for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
                    for (unsigned int r = 0; r < time_degree + 1; ++r)
                    {
                    old_solution_values(q_space + q_time*no_q_space) += old_solution_union.block(r)(local_dof_indices_space[i])*fe_values_space_union.shape_value(i,q_space)*old_fe_values_time.shape_value(r,q_time);
                    solution_values(q_space + q_time*no_q_space) += solution_union.block(r)(local_dof_indices_space[i])*fe_values_space_union.shape_value(i,q_space)*fe_values_time.shape_value(r,q_time);
                    }

    fe_values_space_union.get_function_values (old_old_solution_plus_union, old_old_solution_plus_values);

        for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
        {
        switch (time_degree) {case 0: L2_projection_f(time_degree) = old_solution_values(q_space)*old_solution_values(q_space); break;
        default: L2_projection_f(time_degree) = 0; L2_projection_rhs = 0; solution_time_derivative_value = 0;

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            nonlinearity_value = old_solution_values(q_space + q_time*no_q_space)*old_solution_values(q_space + q_time*no_q_space)*old_fe_values_time.JxW(q_time);

	            for (unsigned int i = 0; i < time_degree + 1; ++i)
	            L2_projection_rhs(i) += nonlinearity_value*old_fe_values_time.shape_value(i, q_time);
            }

            for (unsigned int i = 0; i < time_degree + 1; ++i)
            L2_projection_f(time_degree) += old_temporal_mass_matrix_inv(time_degree, i)*L2_projection_rhs(i);

	        for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
                for (unsigned int r = 0; r < time_degree + 1; ++r)
                solution_time_derivative_value += old_solution_union.block(r)(local_dof_indices_space[i])*fe_values_space_union.shape_value(i, q_space)*old_fe_values_time.shape_grad(r, no_q_time - 1)[0];
        }

        discrete_laplacian_jump_value = -L2_projection_f(time_degree) + solution_time_derivative_value + (1/dt_old)*Q_derivative_values[no_q_time-1]*(old_solution_values(q_space) - old_old_solution_plus_values[q_space]);

        switch (time_degree) {case 0: for (unsigned int q_time = 0; q_time < no_q_time; ++q_time) {L2_projection_f_values[q_time] = solution_values(q_space)*solution_values(q_space);} break;
        default: L2_projection_rhs = 0; solution_time_derivative_value = 0;

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            nonlinearity_value = solution_values(q_space + q_time*no_q_space)*solution_values(q_space + q_time*no_q_space)*fe_values_time.JxW(q_time);

                for (unsigned int i = 0; i < time_degree + 1; ++i)
                L2_projection_rhs(i) += nonlinearity_value*fe_values_time.shape_value(i, q_time);
            }

        temporal_mass_matrix_inv.vmult (L2_projection_f, L2_projection_rhs);
        fe_values_time.get_function_values (L2_projection_f, L2_projection_f_values);

            for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
                for (unsigned int r = 0; r < time_degree + 1; ++r)
                solution_time_derivative_value += solution_union.block(r)(local_dof_indices_space[i])*fe_values_space_union.shape_value(i, q_space)*fe_values_time.shape_grad(r, 0)[0];
        }

        jump_value = solution_values(q_space) - old_solution_values(q_space + (no_q_time - 1)*no_q_space);
        discrete_laplacian_jump_value += L2_projection_f_values[0] - solution_time_derivative_value - (1/dt)*Q_derivative_values[0]*jump_value;

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            double f_reconstructed = solution_values(q_space + q_time*no_q_space) + Q_values[q_time]*jump_value; f_reconstructed *= f_reconstructed;
            estimator_values(q_time) = fmax(estimator_values(q_time), fabs(f_reconstructed - L2_projection_f_values[q_time] - Q_values[q_time]*discrete_laplacian_jump_value));
            }
        }
    }
}

    // Integrate the Linfty norm of the temporal residual at the temporal quadrature points to get the time estimator

    for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
    time_est += estimator_values(q_time)*fe_values_time.JxW(q_time);
}

// Solves the delta equation to determine if the estimator can be computed and, if it can be, computes it and outputs it along with other values of interest

template <int dim> void dGcGblowup<dim>::compute_estimator ()
{
// Try to solve the delta equation via Newton iteration

est += space_est + time_est; delta += 0.5;

delta_residual = 1 + delta*(2*solution_integral - 1) + 2*dt*est*delta*delta;

    for (unsigned int i = 0; i < 10; ++i)
    {
    delta += -delta_residual/(2*solution_integral - 1 + 4*dt*est*delta);
    delta_residual = 1 + delta*(2*solution_integral - 1) + 2*dt*est*delta*delta;
    if (fabs(delta_residual) < 1e-15) {break;}
    }

r = exp(2*solution_integral + dt*delta*est); est *= r;

if (fabs(delta_residual) > delta_residual_threshold)
{
deallog << std::endl << "No solution to the delta equation found -- aborting!" << std::endl;
}
else
{
deallog << std::endl << "max||U(t)||: " << solution.block(time_degree).linfty_norm() << std::endl; // Output a (crude) approximation to the LinftyLinfty norm of the numerical solution
deallog << "Estimator: " << est << std::endl; // Output the value of the estimator
deallog << "Space Estimator: " << space_est << std::endl; // Output the value of the space estimator
deallog << "Time Estimator: " << time_est << std::endl; // Output the value of the time estimator
deallog << "r: " << r << std::endl << std::endl; // Output the value of the scaling parameter r_m
}
}

template <int dim> void dGcGblowup<dim>::run ()
{
deallog << "Spatial Polynomial Degree: " << space_degree << std::endl;
deallog << "Temporal Polynomial Degree: " << time_degree << std::endl;

//GridGenerator::hyper_cube (triangulation_space, -5, 5); triangulation_space.refine_global (2);
GridGenerator::hyper_cube (triangulation_space, -10, 10); triangulation_space.refine_global (2);

// Refine the mesh based on the initial condition
deallog << std::endl << "~~Refining the mesh based on the initial condition~~" << std::endl;

refine_initial_mesh ();

// Setup meshes
old_triangulation_space.copy_triangulation (triangulation_space); old_old_triangulation_space.copy_triangulation (triangulation_space);
GridGenerator::hyper_cube (triangulation_time, 0, dt); old_triangulation_time.copy_triangulation (triangulation_time);

deallog << std::endl << "~~Setting up the initial mesh and timestep length on the first timestep~~" << std::endl;

    for (; fabs(delta_residual) < delta_residual_threshold; ++timestep_number) // Continue computing until the delta equation no longer has a solution
    {
    if (timestep_number == 0)
    {
    while (mesh_change == true)
    {
    mesh_change = false;

    setup_system_full ();

    deallog << std::endl << "Total Degrees of Freedom: " << dof_handler.n_dofs () << std::endl;
    deallog << "Spatial Degrees of Freedom: " << dof_handler_space.n_dofs() << std::endl;
    deallog << "\u0394t: " << dt << std::endl << std::endl;
    deallog << "Projecting the initial condition..." << std::endl;

    energy_project (2*space_degree + 1, initialvalueslaplacian<dim>(), old_solution.block(time_degree)); 
    output_solution ();
    assemble_and_solve (int(1.5*space_degree) + 1, int(1.5*time_degree) + 1); // Setup and solve the system and output the numerical solution
    compute_space_estimator (int(1.5*space_degree) + 1, int(1.5*time_degree) + 2, true); // Compute the space estimator
    compute_time_estimator (int(1.5*space_degree) + 1, int(1.5*time_degree) + 2); // Compute the time estimator

    deallog << std::endl << "Space Estimator: " << space_est << std::endl; // Output the value of the time estimator
    deallog << "Time Estimator: " << time_est << std::endl; // Output the value of the time estimator

    refine_mesh ();

    if (time_est > temporal_refinement_threshold)
    {
    dt = 0.5*dt; dt_old = dt; triangulation_time.clear(); GridGenerator::hyper_cube (triangulation_time, 0, dt); old_triangulation_time.clear(); old_triangulation_time.copy_triangulation (triangulation_time);
    mesh_change = true;
    }
    if (mesh_change == true) {deallog << std::endl << "Estimators are too large. Refining the initial mesh and/or timestep length..." << std::endl;} else {deallog << std::endl << "Estimators are sufficiently small. Proceeding to the first timestep." << std::endl;}
    }
    }
    else
    {
    assemble_and_solve (int(1.5*space_degree) + 1, int(1.5*time_degree) + 1); // Setup and solve the system and output the numerical solution
    compute_space_estimator (int(1.5*space_degree) + 1, int(1.5*time_degree) + 2, true); // Compute the space estimator
    compute_time_estimator (int(1.5*space_degree) + 1, int(1.5*time_degree) + 2); // Compute the time estimator

    refine_mesh ();

    if (time_est > temporal_refinement_threshold)
    {
    dt = 0.5*dt; triangulation_time.clear(); GridGenerator::hyper_cube (triangulation_time, 0, dt);
    }

    if (mesh_change == true || time_est > temporal_refinement_threshold)
    {
    deallog << std::endl;
    if (mesh_change == true) {deallog << "The mesh has changed. ";}
    if (time_est > temporal_refinement_threshold) {deallog << "The time step length has changed. ";}
    deallog << "Recomputing the solution..." << std::endl << std::endl;

    setup_system_partial ();
    assemble_and_solve (int(1.5*space_degree) + 1, int(1.5*time_degree) + 1); // Setup and solve the system and output the numerical solution
    compute_space_estimator (int(1.5*space_degree) + 1, int(1.5*time_degree) + 2, false); // Compute the space estimator
    compute_time_estimator (int(1.5*space_degree) + 1, int(1.5*time_degree) + 2); // Compute the time estimator
    }

    }

    if (timestep_number == 0) {timestep_number = 1;}

    time = time + dt;

    deallog  << std::endl << "Timestep " << timestep_number << " at t=" << std::setprecision (8) << time << std::setprecision (6) << std::endl;
    deallog << "Total Degrees of Freedom: " << dof_handler.n_dofs () << std::endl;
    deallog << "Spatial Degrees of Freedom: " << dof_handler_space.n_dofs () << std::endl;
    deallog << "\u0394t: " << dt << std::endl;

    output_solution ();
    compute_estimator ();

    prepare_for_next_time_step ();
    }
}

int main ()
{
deallog.depth_console (2); std::ofstream logfile ("deallog"); deallog.attach (logfile);

try 
{
dGcGblowup<2> dGcG; dGcG.run ();
}
catch (std::exception &exc)
{
std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl << "Exception on processing: " << std::endl << exc.what() << std::endl << "Aborting!" << std::endl << "----------------------------------------------------" << std::endl;
return 1;
}
catch (...)
{
std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl << "Unknown exception!" << std::endl << "Aborting!" << std::endl << "----------------------------------------------------" << std::endl;
return 1;
};

return 0;
}