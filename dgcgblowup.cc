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

    //values[point] = 10*(x*x + y*y)*exp(-0.5*(x*x + y*y));
    values[point] = 10*exp(-2*(x*x + y*y));
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

    //values[point] = 10*(x*x*x*x + y*y*y*y + 2*x*x*y*y - 6*x*x - 6*y*y + 4)*exp(-0.5*(x*x + y*y));
    values[point] = 80*(2*(x*x + y*y) - 1)*exp(-2*(x*x + y*y));
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
    const unsigned int space_degree = 3; // Spatial polynomial degree
	const unsigned int time_degree = 1; // Temporal polynomial degree
    unsigned int timestep_number = 0; // The current timestep
    double time = 0; // The current time
    double dt = 0.1*0.215; // The current timestep length
	double dt_old = dt; // The timestep length on the last time interval

	// Error estimator parameters
	double estimator = 0; // The error estimator
	double etaS = 10; // The space estimator
	double etaT = 0; // The time estimator
	double r = 0; // The scaling parameter r_m
	double delta = 1.5; // The scaling parameter delta_m (the solution of the delta equation)
	double solution_time_integral = 0; // The (time) integral of the Linfty norm of the numerical solution
	double delta_residual = 0; // The residual arising from the numerical solution of the delta equation

	// Error estimator thresholds
    double spatial_refinement_threshold = 0.1; // The spatial refinement threshold
    double spatial_coarsening_threshold = 0.1*std::pow(2.0, -1.0*space_degree)*spatial_refinement_threshold; // The spatial coarsening threshold
	double temporal_refinement_threshold = 1e-3; // The temporal refinement threshold
	double delta_residual_threshold = 1e-04; // The threshold for the delta equation residual above which we consider the delta equation as having no root

    // Mesh change parameters
    bool mesh_change = true; // Parameter indicating if mesh change recently occured

private:

    void setup_system_full ();
	void setup_system_time ();
	void create_system_matrix ();
    void create_temporal_mass_matrix (const FE_DGQ<1> &fe_time, FullMatrix<double> &temporal_mass_matrix) const;
	void create_time_derivative_matrix (const FE_DGQ<1> &fe_time, FullMatrix<double> &time_derivative_matrix) const;
    void energy_project (const unsigned int &no_q_space_x, const Function<dim> &laplacian_function, Vector<double> &projection) const;
	void assemble_and_solve (const unsigned int &no_q_space_x, const unsigned int &no_q_time, const unsigned int &max_iterations, const double &rel_tol);
    void refine_initial_mesh (); 
    void refine_mesh ();
	void output_solution () const; // Output the solution
	void get_spacetime_function_values (const Vector<double> &spacetime_fe_function, const FEValues<dim> &fe_values_space, const FEValues<1> &fe_values_time, const std::vector<types::global_dof_index> &local_dof_indices, Vector<double> &spacetime_fe_function_values) const;
	void reorder_solution_vector (const Vector<double> &spacetime_fe_function, BlockVector<double> &reordered_spacetime_fe_function) const;
	void extend_to_constant_in_time_function (Vector<double> &fe_function, Vector<double> &spacetime_fe_function) const;
	void compute_Q_values (const unsigned int &degree, const double &point, double &Q_value, double &Q_derivative_value, double &Q_second_derivative_value) const;
	void compute_space_estimator (const unsigned int &no_q_space_x, const unsigned int &no_q_time);
	void compute_time_estimator (const unsigned int &no_q_space_x, const unsigned int &no_q_time);
	void compute_estimator ();

	Triangulation<dim> triangulation_space; Triangulation<dim> old_triangulation_space; Triangulation<dim> old_old_triangulation_space; 
	Triangulation<1> triangulation_time; Triangulation<1> old_triangulation_time;

	DoFHandler<dim> dof_handler_space; DoFHandler<dim> old_dof_handler_space; DoFHandler<dim> old_old_dof_handler_space; 
	DoFHandler<1> dof_handler_time; DoFHandler<1> old_dof_handler_time; 
	DoFHandler<dim> dof_handler; DoFHandler<dim> old_dof_handler;

    FE_Q<dim> fe_space; FE_Q<dim> old_fe_space; FE_Q<dim> old_old_fe_space; 
	FE_DGQ<1> fe_time; FE_DGQ<1> old_fe_time; 
	FESystem<dim> fe; FESystem<dim> old_fe;

	ConstraintMatrix constraints;
	SparsityPattern sparsity_pattern;

	SparseMatrix<double> system_matrix;

	BlockVector<double> reordered_solution;

	Vector<double> right_hand_side;
	Vector<double> solution;
	Vector<double> old_solution;
	Vector<double> solution_plus;
	Vector<double> old_solution_plus;
	Vector<double> old_old_solution_plus;
    Vector<double> refinement_vector;
};

template <int dim> dGcGblowup<dim>::dGcGblowup ()
                :
				dof_handler_space (triangulation_space), old_dof_handler_space (old_triangulation_space), old_old_dof_handler_space (old_old_triangulation_space),
				dof_handler_time (triangulation_time), old_dof_handler_time (old_triangulation_time),
				dof_handler (triangulation_space), old_dof_handler (old_triangulation_space),
				fe_space (space_degree), old_fe_space (space_degree), old_old_fe_space (space_degree),
				fe_time (time_degree), old_fe_time (time_degree),
				fe (fe_space, time_degree + 1), old_fe (old_fe_space, time_degree + 1)
{}

template <int dim> void dGcGblowup<dim>::setup_system_full ()
{
dof_handler_space.distribute_dofs (fe_space); old_dof_handler_space.distribute_dofs (old_fe_space); old_old_dof_handler_space.distribute_dofs (old_old_fe_space);
dof_handler_time.distribute_dofs (fe_time); old_dof_handler_time.distribute_dofs (old_fe_time);
dof_handler.distribute_dofs (fe); old_dof_handler.distribute_dofs (old_fe); 

unsigned int no_of_space_dofs = dof_handler_space.n_dofs ();
unsigned int no_of_old_space_dofs = old_dof_handler_space.n_dofs ();
unsigned int no_of_old_old_space_dofs = old_old_dof_handler_space.n_dofs ();
unsigned int no_of_dofs = no_of_space_dofs*(time_degree + 1);
unsigned int no_of_old_dofs = no_of_old_space_dofs*(time_degree + 1);
unsigned int no_of_cells = triangulation_space.n_active_cells ();

constraints.clear ();
DoFTools::make_hanging_node_constraints (dof_handler, constraints);
DoFTools::make_zero_boundary_constraints (dof_handler, constraints);
constraints.close ();

DynamicSparsityPattern dsp (no_of_dofs);
DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
sparsity_pattern.copy_from (dsp);

system_matrix.reinit (sparsity_pattern);

reordered_solution.reinit (time_degree + 1);
for (unsigned int i = 0; i < time_degree + 1; ++i)
{
reordered_solution.block(i).reinit (no_of_space_dofs);
}
reordered_solution.collect_sizes ();

right_hand_side.reinit (no_of_dofs);
solution.reinit (no_of_dofs);
old_solution.reinit (no_of_old_dofs);
solution_plus.reinit (no_of_space_dofs);
old_solution_plus.reinit (no_of_old_space_dofs);
old_old_solution_plus.reinit (no_of_old_old_space_dofs);
refinement_vector.reinit (no_of_cells);

create_system_matrix ();
}

template <int dim> void dGcGblowup<dim>::setup_system_time ()
{
dof_handler_time.distribute_dofs (fe_time);

system_matrix.reinit (sparsity_pattern);

create_system_matrix ();
}

template <int dim> void dGcGblowup<dim>::create_system_matrix ()
{
const QGauss<dim> quadrature_formula_space (space_degree + 1);

FEValues<dim> fe_values_space (fe_space, quadrature_formula_space, update_values | update_gradients | update_JxW_values);

const unsigned int no_q_space = quadrature_formula_space.size ();
const unsigned int dofs_per_cell_space = fe_space.dofs_per_cell; const unsigned int dofs_per_cell = fe.dofs_per_cell;

FullMatrix<double> local_system_matrix (dofs_per_cell, dofs_per_cell);
FullMatrix<double> local_mass_matrix (dofs_per_cell_space, dofs_per_cell_space);
FullMatrix<double> local_laplace_matrix (dofs_per_cell_space, dofs_per_cell_space);
FullMatrix<double> temporal_mass_matrix (time_degree + 1, time_degree + 1);
FullMatrix<double> time_derivative_matrix (time_degree + 1, time_degree + 1);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

create_temporal_mass_matrix (fe_time, temporal_mass_matrix);
if (time_degree > 0) {create_time_derivative_matrix (fe_time, time_derivative_matrix);}

typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active (), final_space_cell = dof_handler_space.end ();
typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active ();

double cell_size = 0; double previous_cell_size = 0; double cell_size_check = 0;

    for (; space_cell != final_space_cell; ++cell, ++space_cell)
    {
	cell->get_dof_indices (local_dof_indices);
    cell_size = space_cell->measure ();

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
        unsigned int comp_s_k = fe.system_to_component_index(k).second; unsigned int comp_t_k = fe.system_to_component_index(k).first;

            for (unsigned int l = 0; l < dofs_per_cell; ++l)
            {
            unsigned int comp_s_l = fe.system_to_component_index(l).second; unsigned int comp_t_l = fe.system_to_component_index(l).first;

            switch(time_degree)
            {
            case 0: local_system_matrix(k,l) += temporal_mass_matrix(comp_t_k, comp_t_l)*local_laplace_matrix(comp_s_k, comp_s_l); break;
            default: local_system_matrix(k,l) += time_derivative_matrix(comp_t_k, comp_t_l)*local_mass_matrix(comp_s_k, comp_s_l) + temporal_mass_matrix(comp_t_k, comp_t_l)*local_laplace_matrix(comp_s_k, comp_s_l);
            }

            if ((comp_t_k == 0) && (comp_t_l == 0)) {local_system_matrix(k,l) += local_mass_matrix(comp_s_k, comp_s_l);}
            }
        }
    }

    constraints.distribute_local_to_global (local_system_matrix, local_dof_indices, system_matrix);
    previous_cell_size = cell_size; 
    }
}

template <int dim> void dGcGblowup<dim>::create_temporal_mass_matrix (const FE_DGQ<1> &fe_time, FullMatrix<double> &temporal_mass_matrix) const
{
const QGauss<1> quadrature_formula_time (time_degree + 1);
FEValues<1> fe_values_time (fe_time, quadrature_formula_time, update_values | update_JxW_values);

const unsigned int no_q_time = quadrature_formula_time.size ();

typename DoFHandler<1>::active_cell_iterator time_cell = dof_handler_time.begin_active (); fe_values_time.reinit (time_cell);

    for (unsigned int r = 0; r < time_degree + 1; ++r)
        for (unsigned int s = 0; s < r + 1; ++s)
        {
            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
	        temporal_mass_matrix(r,s) += fe_values_time.shape_value(r,q_time)*fe_values_time.shape_value(s,q_time)*fe_values_time.JxW(q_time);
            }
   
		temporal_mass_matrix(s,r) = temporal_mass_matrix(r,s);
        }
}

template <int dim> void dGcGblowup<dim>::create_time_derivative_matrix (const FE_DGQ<1> &fe_time, FullMatrix<double> &time_derivative_matrix) const
{
const QGauss<1> quadrature_formula_time (time_degree + 1);
FEValues<1> fe_values_time (fe_time, quadrature_formula_time, update_values | update_gradients | update_JxW_values);

const unsigned int no_q_time = quadrature_formula_time.size ();

typename DoFHandler<1>::active_cell_iterator time_cell = dof_handler_time.begin_active (); fe_values_time.reinit (time_cell);

    for (unsigned int r = 0; r < time_degree + 1; ++r)
        for (unsigned int s = 0; s < time_degree + 1; ++s)
            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
	        time_derivative_matrix(r,s) += fe_values_time.shape_value(r,q_time)*fe_values_time.shape_grad(s,q_time)[0]*fe_values_time.JxW(q_time);
            }
}

template <int dim> void dGcGblowup<dim>::energy_project (const unsigned int &no_q_space_x, const Function<dim> &laplacian_function, Vector<double> &projection) const
{
const QGauss<dim> quadrature_formula_space (no_q_space_x);

FEValues<dim> fe_values_space (fe_space, quadrature_formula_space, update_values | update_gradients | update_quadrature_points | update_JxW_values);

const unsigned int no_q_space = quadrature_formula_space.size ();
const unsigned int no_of_space_dofs = dof_handler_space.n_dofs ();
const unsigned int dofs_per_cell_space = fe_space.dofs_per_cell;

ConstraintMatrix spatial_constraints; SparsityPattern spatial_sparsity_pattern;

spatial_constraints.clear ();
DoFTools::make_hanging_node_constraints (dof_handler_space, spatial_constraints);
DoFTools::make_zero_boundary_constraints (dof_handler_space, spatial_constraints);
spatial_constraints.close ();

DynamicSparsityPattern dsp (no_of_space_dofs);
DoFTools::make_sparsity_pattern (dof_handler_space, dsp);
spatial_constraints.condense (dsp);
spatial_sparsity_pattern.copy_from (dsp);

SparseMatrix<double> laplace_matrix; laplace_matrix.reinit (spatial_sparsity_pattern);
FullMatrix<double> local_laplace_matrix (dofs_per_cell_space, dofs_per_cell_space);
Vector<double> right_hand_side (no_of_space_dofs);
Vector<double> local_right_hand_side (dofs_per_cell_space);
std::vector<double> laplacian_values (no_q_space);
std::vector<types::global_dof_index> local_dof_indices_space (dofs_per_cell_space);

typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active (), final_space_cell = dof_handler_space.end ();

    for (; space_cell != final_space_cell; ++space_cell)
    {
    local_laplace_matrix = 0; local_right_hand_side = 0;
    fe_values_space.reinit (space_cell);
    space_cell->get_dof_indices (local_dof_indices_space);

    laplacian_function.value_list (fe_values_space.get_quadrature_points(), laplacian_values);

        for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
        {
            for (unsigned int j = 0; j < i + 1; ++j)
            {
                for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
                {
                local_laplace_matrix(i,j) += fe_values_space.shape_grad(i,q_space)*fe_values_space.shape_grad(j,q_space)*fe_values_space.JxW(q_space);
                }
             
            local_laplace_matrix(j,i) = local_laplace_matrix(i,j);
            }

            for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
            {
            local_right_hand_side(i) -= laplacian_values[q_space]*fe_values_space.shape_value(i,q_space)*fe_values_space.JxW(q_space);
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell_space; ++i)
        {
            for (unsigned int j = 0; j < dofs_per_cell_space; ++j)
            {
            laplace_matrix (local_dof_indices_space[i], local_dof_indices_space[j]) += local_laplace_matrix(i,j);
            }

        right_hand_side(local_dof_indices_space[i]) += local_right_hand_side(i);    
        }
    }

SolverBicgstab<>::AdditionalData data; data.exact_residual = false;

SolverControl solver_control (10000, 1e-20, false, false);
SolverBicgstab<> solver (solver_control, data);

spatial_constraints.condense (laplace_matrix, right_hand_side);

SparseILU<double> ilu; ilu.initialize (laplace_matrix);
solver.solve (laplace_matrix, projection, right_hand_side, ilu);

spatial_constraints.distribute (projection);
}

template <int dim> void dGcGblowup<dim>::assemble_and_solve (const unsigned int &no_q_space_x, const unsigned int &no_q_time, const unsigned int &max_iterations, const double &rel_tol)
{
deallog << "Calculating the numerical solution via Picard iteration..." << std::endl;

const QGauss<dim> quadrature_formula_space (no_q_space_x); const QGauss<1> quadrature_formula_time (no_q_time);

FEValues<dim> fe_values_space (fe_space, quadrature_formula_space, update_values | update_quadrature_points | update_JxW_values);
FEValues<1> fe_values_time (fe_time, quadrature_formula_time, update_values | update_JxW_values);

const unsigned int no_q_space = quadrature_formula_space.size();
const unsigned int dofs_per_cell = fe.dofs_per_cell;

std::vector<double> old_solution_plus_values (no_q_space);
Vector<double> nonlinearity_values (no_q_space*no_q_time);
Vector<double> residual_vector (dof_handler.n_dofs());
Vector<double> local_right_hand_side (dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

Functions::FEFieldFunction<dim> old_solution_plus_function (old_dof_handler_space, old_solution_plus);

if (mesh_change == false || timestep_number == 0)
{
switch (time_degree)
{
case 0: solution = old_solution_plus; break;
default: extend_to_constant_in_time_function (old_solution_plus, solution);
}
}
else
{
VectorTools::interpolate_to_different_mesh (old_dof_handler_space, old_solution_plus, dof_handler_space, solution_plus);

switch (time_degree)
{
case 0: solution = solution_plus; break;
default: extend_to_constant_in_time_function (solution_plus, solution);
}
}

typename DoFHandler<1>::active_cell_iterator time_cell = dof_handler_time.begin_active (); fe_values_time.reinit (time_cell);

unsigned int iteration_number = 1; double residual = 0; double max = solution.linfty_norm ();

    for (; iteration_number < max_iterations; ++iteration_number)
    {
    typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active ();
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active (), final_cell = dof_handler.end ();

    residual_vector = solution;

    right_hand_side = 0;

        for (; cell != final_cell; ++cell, ++space_cell)
        {
        local_right_hand_side = 0;
        fe_values_space.reinit (space_cell);

        cell->get_dof_indices (local_dof_indices);

        old_solution_plus_function.value_list (fe_values_space.get_quadrature_points(), old_solution_plus_values); 
        get_spacetime_function_values (solution, fe_values_space, fe_values_time, local_dof_indices, nonlinearity_values);

            for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
                for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
                {
                nonlinearity_values(q_space + q_time*no_q_space) *= nonlinearity_values(q_space + q_time*no_q_space)*fe_values_space.JxW(q_space)*fe_values_time.JxW(q_time);
                } 

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
            unsigned int comp_s_i = fe.system_to_component_index(i).second; unsigned int comp_t_i = fe.system_to_component_index(i).first;

                for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
                {
                    for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
	                {
	                local_right_hand_side(i) += nonlinearity_values(q_space + q_time*no_q_space)*fe_values_space.shape_value(comp_s_i,q_space)*fe_values_time.shape_value(comp_t_i,q_time);
 	                }

                if (comp_t_i == 0) {local_right_hand_side(i) += old_solution_plus_values[q_space]*fe_values_space.shape_value(comp_s_i,q_space)*fe_values_space.JxW(q_space);}
                }
            }

        constraints.distribute_local_to_global (local_right_hand_side, local_dof_indices, right_hand_side);
        }

    SolverBicgstab<>::AdditionalData data; data.exact_residual = false;

    SolverControl solver_control (1000, max*sqrt(max)*rel_tol*sqrt(rel_tol), false, false);
    SolverBicgstab<> solver (solver_control, data);

    SparseILU<double> ilu; ilu.initialize (system_matrix);
    solver.solve (system_matrix, solution, right_hand_side, ilu);

    constraints.distribute (solution);

    residual_vector.add (-1,solution);
    residual = residual_vector.l2_norm ();

    if (residual < max*rel_tol) {break;}
    }

switch(time_degree) {case 0: solution_plus = solution; break; default: reorder_solution_vector (solution, reordered_solution); solution_plus = reordered_solution.block(time_degree);}

if (iteration_number == max_iterations) {deallog << "...converged in the maximum number of allowed iterations (" << max_iterations << ") with a residual of " << residual << std::endl;} else {deallog << "...converged in " << iteration_number << " iterations with a residual of " << residual << std::endl;}
}

// Refine the mesh

template <int dim> void dGcGblowup<dim>::refine_initial_mesh ()
{
while (etaS > spatial_coarsening_threshold)
{
dof_handler_space.distribute_dofs (fe_space);

Vector<double> projection (dof_handler_space.n_dofs()); Vector<double> error (triangulation_space.n_active_cells());

energy_project (2*space_degree + 1, initialvalueslaplacian<dim>(), projection);

deallog << std::endl << "Spatial Degrees of Freedom: " << dof_handler_space.n_dofs() << std::endl;
deallog << "Projecting the initial condition..." << std::endl;

VectorTools::integrate_difference (dof_handler_space, projection, initialvalues<dim>(), error, QGauss<dim>(int((3*space_degree + 3)/2)), VectorTools::Linfty_norm);
etaS = error.linfty_norm ();

deallog << "Initial Linfty Error: " << etaS << std::endl;

GridRefinement::refine (triangulation_space, error, spatial_coarsening_threshold);
triangulation_space.prepare_coarsening_and_refinement (); triangulation_space.execute_coarsening_and_refinement ();
}	
}

// Refine the mesh

template <int dim> void dGcGblowup<dim>::refine_mesh ()
{
GridRefinement::refine (triangulation_space, refinement_vector, spatial_refinement_threshold);
GridRefinement::coarsen (triangulation_space, refinement_vector, spatial_coarsening_threshold);

triangulation_space.prepare_coarsening_and_refinement (); triangulation_space.execute_coarsening_and_refinement ();

typename Triangulation<dim>::active_cell_iterator space_cell = triangulation_space.begin_active (), final_cell = triangulation_space.end ();
typename Triangulation<dim>::active_cell_iterator old_space_cell = old_triangulation_space.begin_active ();

    for (; space_cell != final_cell; ++space_cell, ++old_space_cell)
    {
        for (unsigned int vertex = 0; vertex < 4; ++vertex)
        {
        if ((space_cell->vertex(0) - old_space_cell->vertex(0))*(space_cell->vertex(0) - old_space_cell->vertex(0)) + (space_cell->vertex(1) - old_space_cell->vertex(1))*(space_cell->vertex(1) - old_space_cell->vertex(1)) > 1e-15) {mesh_change = true; break;}
        }

    if (mesh_change == true) {break;}
    }

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

// Output the solution

template <int dim> void dGcGblowup<dim>::output_solution () const
{
DataOut<dim> data_out; data_out.attach_dof_handler (dof_handler_space); data_out.add_data_vector (solution_plus, "u_h"); data_out.build_patches ();

const std::string filename = "solution-" + Utilities::int_to_string (timestep_number, 5) + ".gnuplot";

std::ofstream gnuplot_output (filename.c_str()); data_out.write_gnuplot (gnuplot_output);
}

template<int dim> void dGcGblowup<dim>::get_spacetime_function_values (const Vector<double> &spacetime_fe_function, const FEValues<dim> &fe_values_space, const FEValues<1> &fe_values_time, const std::vector<types::global_dof_index> &local_dof_indices, Vector<double> &spacetime_fe_function_values) const
{
const unsigned int no_q_space = fe_values_space.get_quadrature().size(); const unsigned int no_q_time = fe_values_time.get_quadrature().size();
const unsigned int dofs_per_cell = fe.dofs_per_cell;

spacetime_fe_function_values = 0;

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
    const unsigned int comp_s_i = fe.system_to_component_index(i).second; const unsigned int comp_t_i = fe.system_to_component_index(i).first;

    double fe_value_i = spacetime_fe_function(local_dof_indices[i]);

        for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            spacetime_fe_function_values (q_space + q_time*no_q_space) += fe_value_i*fe_values_space.shape_value(comp_s_i, q_space)*fe_values_time.shape_value(comp_t_i, q_time);
            }
    }
}

template <int dim> void dGcGblowup<dim>::reorder_solution_vector (const Vector<double> &spacetime_fe_function, BlockVector<double> &reordered_spacetime_fe_function) const
{
const unsigned int dofs_per_cell = fe.dofs_per_cell;

std::vector<types::global_dof_index> local_dof_indices_space (fe_space.dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active ();
typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active (), final_cell = dof_handler.end ();

    for (; cell != final_cell; ++cell, ++space_cell)
    {
    space_cell->get_dof_indices (local_dof_indices_space);
    cell->get_dof_indices (local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
        const unsigned int comp_s_i = fe.system_to_component_index(i).second; const unsigned int comp_t_i = fe.system_to_component_index(i).first;

        reordered_spacetime_fe_function.block(comp_t_i)(local_dof_indices_space[comp_s_i]) = spacetime_fe_function(local_dof_indices[i]);
        }
    }
}

template <int dim> void dGcGblowup<dim>::extend_to_constant_in_time_function (Vector<double> &fe_function, Vector<double> &spacetime_fe_function) const
{
const unsigned int dofs_per_cell = fe.dofs_per_cell;

std::vector<types::global_dof_index> local_dof_indices_space (fe_space.dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active ();
typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active (), final_cell = dof_handler.end ();

    for (; cell != final_cell; ++cell, ++space_cell)
    {
    space_cell->get_dof_indices (local_dof_indices_space);
    cell->get_dof_indices (local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
        const unsigned int comp_s_i = fe.system_to_component_index(i).second;

        spacetime_fe_function(local_dof_indices[i]) = fe_function(local_dof_indices_space[comp_s_i]);
        }
    }
}

template <int dim> void dGcGblowup<dim>::compute_Q_values (const unsigned int &degree, const double &point, double &Q_value, double &Q_derivative_value, double &Q_second_derivative_value) const
{
switch(degree)
{
case 0: Q_value = point - 1; Q_derivative_value = 1; Q_second_derivative_value = 0; break;
case 1: Q_value = 1.5*point*point - point - 0.5; Q_derivative_value = 3*point - 1; Q_second_derivative_value = 3; break;
default: double value = 0; double old_value = point; double old_old_value = 1.0; double derivative_value = 0; double old_derivative_value = 1; double second_derivative_value = 0; double old_second_derivative_value = 0;
    for (unsigned int n = 2; n < degree + 2; ++n)
    {
    value = ((2*n - 1)*point*old_value - (n - 1)*old_old_value)/n;
    derivative_value = point*old_derivative_value + n*old_value;
    second_derivative_value = (n + 1)*old_derivative_value + point*old_second_derivative_value;
    old_old_value = old_value; old_value = value; old_derivative_value = derivative_value; old_second_derivative_value = second_derivative_value;
    if (n == degree) {Q_value = -value; Q_derivative_value = -derivative_value; Q_second_derivative_value = -second_derivative_value;} if (n == degree + 1) {Q_value += value; Q_derivative_value += derivative_value; Q_second_derivative_value += second_derivative_value;}
    }
}
Q_value *= 0.5*std::pow(-1, degree); Q_derivative_value *= std::pow(-1, degree); Q_second_derivative_value *= 2*std::pow(-1, degree);
}

template <int dim> void dGcGblowup<dim>::compute_space_estimator (const unsigned int &no_q_space_x, const unsigned int &no_q_time)
{
const QGaussLobatto<dim> quadrature_formula_space (no_q_space_x); const QGaussLobatto<dim-1> quadrature_formula_space_face (no_q_space_x); const QGaussLobatto<1> quadrature_formula_time (no_q_time);

FEValues<dim> fe_values_space (fe_space, quadrature_formula_space, update_values | update_hessians | update_quadrature_points);
FEFaceValues<dim> fe_values_space_face (fe_space, quadrature_formula_space_face, update_gradients);
FEFaceValues<dim> fe_values_space_face_neighbor (fe_space, quadrature_formula_space_face, update_gradients | update_normal_vectors);
FESubfaceValues<dim> fe_values_space_subface (fe_space, quadrature_formula_space_face, update_gradients | update_normal_vectors);
FEValues<1> fe_values_time (fe_time, quadrature_formula_time, update_values | update_gradients | update_hessians | update_quadrature_points | update_JxW_values);
FEValues<1> old_fe_values_time (old_fe_time, quadrature_formula_time, update_values | update_gradients | update_JxW_values);

const unsigned int no_of_space_dofs = dof_handler_space.n_dofs();
const unsigned int no_of_cells = triangulation_space.n_active_cells();
const unsigned int no_q_space = quadrature_formula_space.size();
const unsigned int no_q_space_face = quadrature_formula_space_face.size();
const unsigned int dofs_per_cell = fe.dofs_per_cell;

FullMatrix<double> temporal_mass_matrix_inv (time_degree + 1, time_degree + 1);
FullMatrix<double> old_temporal_mass_matrix_inv (time_degree + 1, time_degree + 1);

BlockVector<double> space_estimator_values (no_q_time);
BlockVector<double> space_derivative_estimator_values (no_q_time);
BlockVector<double> reordered_solution_at_temporal_quadrature_points (no_q_time);
BlockVector<double> reordered_solution_time_derivative_at_temporal_quadrature_points (no_q_time);

    for (unsigned int i = 0; i < no_q_time; ++i)
    {
    space_estimator_values.block(i).reinit (no_of_cells);
    space_derivative_estimator_values.block(i).reinit (no_of_cells);
    reordered_solution_at_temporal_quadrature_points.block(i).reinit (no_of_space_dofs);
    reordered_solution_time_derivative_at_temporal_quadrature_points.block(i).reinit (no_of_space_dofs);
    }

space_estimator_values.collect_sizes ();
space_derivative_estimator_values.collect_sizes ();
reordered_solution_at_temporal_quadrature_points.collect_sizes ();
reordered_solution_time_derivative_at_temporal_quadrature_points.collect_sizes ();

Vector<double> estimator_values (no_q_time);
Vector<double> derivative_estimator_values (no_q_time);
Vector<double> solution_values (no_q_space*no_q_time);
Vector<double> old_solution_values (no_q_space*no_q_time);
Vector<double> solution_laplacian_values (no_q_space*no_q_time);
Vector<double> solution_time_derivative_values (no_q_space*no_q_time);
Vector<double> solution_time_derivative_laplacian_values (no_q_space*no_q_time);
Vector<double> solution_second_time_derivative_values (no_q_time);
Vector<double> Q_values (no_q_time);
Vector<double> Q_derivative_values (no_q_time);
Vector<double> Q_second_derivative_values (no_q_time);
Vector<double> L2_projection_rhs (time_degree + 1);
Vector<double> L2_projection_f (time_degree + 1);
std::vector<double> solution_values_temp (no_q_space);
std::vector<double> old_old_solution_plus_values (no_q_space);
std::vector<double> L2_projection_f_values (no_q_time);
std::vector<Tensor<1,1>> L2_projection_f_time_derivative_values (no_q_time);
std::vector<Tensor<2,dim>> solution_hessian_values (no_q_space); 

Vector<double> jump_values (no_q_space_x);
std::vector<Tensor<1,dim> > solution_face_gradient_values (no_q_space_x);
std::vector<Tensor<1,dim> > solution_face_gradient_neighbor_values (no_q_space_x);
std::vector<Tensor<1,dim> > solution_time_derivative_face_gradient_values (no_q_space_x);
std::vector<Tensor<1,dim> > solution_time_derivative_face_gradient_neighbor_values (no_q_space_x);

std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

if (time_degree > 0)
{
create_temporal_mass_matrix (fe_time, temporal_mass_matrix_inv); temporal_mass_matrix_inv.gauss_jordan ();

if (dt == dt_old) {old_temporal_mass_matrix_inv = temporal_mass_matrix_inv;}
else {create_temporal_mass_matrix (old_fe_time, old_temporal_mass_matrix_inv); old_temporal_mass_matrix_inv.gauss_jordan ();}
}

typename DoFHandler<1>::active_cell_iterator time_cell = dof_handler_time.begin_active (); typename DoFHandler<1>::active_cell_iterator old_time_cell = old_dof_handler_time.begin_active ();
fe_values_time.reinit (time_cell); old_fe_values_time.reinit (old_time_cell);

    for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
    {
    compute_Q_values (time_degree, (2/dt)*fe_values_time.quadrature_point(q_time)(0) - 1, Q_values(q_time), Q_derivative_values(q_time), Q_second_derivative_values(q_time));
    }

typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active (), final_cell = dof_handler.end ();
typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active ();

if (time_degree > 0)
{
    for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
        for (unsigned int i = 0; i < no_of_space_dofs; ++i)
	        for (unsigned int j = 0; j < time_degree + 1; ++j)
            {
            reordered_solution_at_temporal_quadrature_points.block(q_time)(i) += reordered_solution.block(j)(i)*fe_values_time.shape_value(j, q_time);
            reordered_solution_time_derivative_at_temporal_quadrature_points.block(q_time)(i) += reordered_solution.block(j)(i)*fe_values_time.shape_grad(j, q_time)[0];
	        }
}

double h = 0; double h_min = GridTools::minimal_cell_diameter (triangulation_space); double ell_h = log(2 + 1/h_min); double C_cell = 0; double C_edge = 0;
etaS = 0; double space_estimator_jump_value = 0; double nonlinearity_value = 0;
refinement_vector = 0;

    for (; cell != final_cell; ++cell, ++space_cell)
    {
    estimator_values = 0; derivative_estimator_values = 0;
    fe_values_space.reinit (space_cell);
    cell->get_dof_indices (local_dof_indices);

    const unsigned int cell_no = cell->active_cell_index ();   
    h = cell->diameter(); C_cell = fmin(1/a, h*h*ell_h/a); C_edge = fmin(1, h*ell_h);

    switch(time_degree)
    {	
    case 0: fe_values_space.get_function_values (solution, solution_values_temp); if (space_degree > 1) {fe_values_space.get_function_hessians (solution, solution_hessian_values);}

        for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
            {
            solution_values (q_space + q_time*no_q_space) = solution_values_temp[q_space]; if (space_degree > 1) {solution_laplacian_values (q_space + q_time*no_q_space) = a*trace(solution_hessian_values[q_space]);}
            }

    break;

    default: std::vector<double> solution_time_derivative_values_temp (no_q_space); std::vector<Tensor<2,dim>> solution_time_derivative_hessian_values (no_q_space);

        for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
        {
        fe_values_space.get_function_values (reordered_solution_at_temporal_quadrature_points.block(q_time), solution_values_temp); fe_values_space.get_function_values (reordered_solution_time_derivative_at_temporal_quadrature_points.block(q_time), solution_time_derivative_values_temp);
        if (space_degree > 1)  {fe_values_space.get_function_hessians (reordered_solution_at_temporal_quadrature_points.block(q_time), solution_hessian_values); fe_values_space.get_function_hessians (reordered_solution_time_derivative_at_temporal_quadrature_points.block(q_time), solution_time_derivative_hessian_values);}

            for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
	        {
            solution_values (q_space + q_time*no_q_space) = solution_values_temp[q_space]; solution_time_derivative_values (q_space + q_time*no_q_space) = solution_time_derivative_values_temp[q_space];
            if (space_degree > 1) {solution_laplacian_values (q_space + q_time*no_q_space) = a*trace(solution_hessian_values[q_space]); solution_time_derivative_laplacian_values(q_space + q_time*no_q_space) = a*trace(solution_time_derivative_hessian_values[q_space]);}
            }
        }
    }

    if (timestep_number > 1) {get_spacetime_function_values (old_solution, fe_values_space, old_fe_values_time, local_dof_indices, old_solution_values); fe_values_space.get_function_values (old_old_solution_plus, old_old_solution_plus_values);}
    else {fe_values_space.get_function_values (old_solution_plus, old_old_solution_plus_values); for (unsigned int q_space = 0; q_space < no_q_space; ++q_space) {old_solution_values (q_space + (no_q_time - 1)*no_q_space) = old_old_solution_plus_values[q_space];} 
    initialvalueslaplacian<dim>().value_list (fe_values_space.get_quadrature_points(), old_old_solution_plus_values); fe_values_space.get_function_hessians (old_solution_plus, solution_hessian_values);}
    
        for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
        {
        space_estimator_jump_value = 0; 

        if (timestep_number > 1)
        {
        space_estimator_jump_value = (1/dt_old)*Q_derivative_values(no_q_time - 1)*(old_solution_values(q_space) - old_old_solution_plus_values[q_space]);

        switch (time_degree) {case 0: space_estimator_jump_value -= old_solution_values(q_space)*old_solution_values(q_space); break;
        default: L2_projection_rhs = 0;

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            nonlinearity_value = old_solution_values(q_space + q_time*no_q_space)*old_solution_values(q_space + q_time*no_q_space)*old_fe_values_time.JxW(q_time);
  
	            for (unsigned int i = 0; i < time_degree + 1; ++i)
                {
                L2_projection_rhs(i) += nonlinearity_value*old_fe_values_time.shape_value(i, q_time);
                }
            }

	        for (unsigned int i = 0; i < time_degree + 1; ++i)
            {
            space_estimator_jump_value -= old_temporal_mass_matrix_inv(time_degree, i)*L2_projection_rhs(i);
            }
        }
  
        if (time_degree > 0 || space_degree > 1)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
            const unsigned int comp_s_i = fe.system_to_component_index(i).second; const unsigned int comp_t_i = fe.system_to_component_index(i).first;

            if (time_degree > 0) {space_estimator_jump_value += old_solution(local_dof_indices[i])*fe_values_space.shape_value(comp_s_i, q_space)*old_fe_values_time.shape_grad(comp_t_i, no_q_time - 1)[0];}
	        if (space_degree > 1) {space_estimator_jump_value -= old_solution(local_dof_indices[i])*a*trace(fe_values_space.shape_hessian(comp_s_i, q_space))*old_fe_values_time.shape_value(comp_t_i, no_q_time - 1);}
            }
        }
        }
        else
        {
        space_estimator_jump_value = a*(old_old_solution_plus_values[q_space] - trace(solution_hessian_values[q_space]));
        } 

        switch(time_degree) {case 0: for (unsigned int q_time = 0; q_time < no_q_time; ++q_time) {L2_projection_f_values[q_time] = solution_values(q_space)*solution_values(q_space);} break;
        default: L2_projection_rhs = 0;

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            nonlinearity_value = solution_values(q_space + q_time*no_q_space)*solution_values(q_space + q_time*no_q_space)*fe_values_time.JxW(q_time);

                for (unsigned int i = 0; i < time_degree + 1; ++i)
                {
                L2_projection_rhs(i) += nonlinearity_value*fe_values_time.shape_value(i, q_time);
                }
            }

        temporal_mass_matrix_inv.vmult (L2_projection_f, L2_projection_rhs);
        fe_values_time.get_function_values (L2_projection_f, L2_projection_f_values);
        fe_values_time.get_function_gradients (L2_projection_f, L2_projection_f_time_derivative_values);
        }

        solution_second_time_derivative_values = 0;

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
   	            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                const unsigned int comp_s_i = fe.system_to_component_index(i).second;
                const unsigned int comp_t_i = fe.system_to_component_index(i).first;

   	            solution_second_time_derivative_values(q_time) += solution(local_dof_indices[i])*fe_values_space.shape_value(comp_s_i, q_space)*fe_values_time.shape_hessian(comp_t_i, q_time)[0][0];
		        }

        space_estimator_jump_value += L2_projection_f_values[0] - solution_time_derivative_values(q_space) - (1/dt)*Q_derivative_values(0)*(solution_values(q_space) - old_solution_values(q_space + (no_q_time - 1)*no_q_space)) + solution_laplacian_values(q_space);

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
	        {
	        estimator_values(q_time) = fmax(estimator_values(q_time), fabs(L2_projection_f_values[q_time] - solution_time_derivative_values(q_space + q_time*no_q_space) - (1/dt)*Q_derivative_values(q_time)*(solution_values(q_space) - old_solution_values(q_space + (no_q_time - 1)*no_q_space)) + solution_laplacian_values(q_space + q_time*no_q_space) + Q_values(q_time)*space_estimator_jump_value)); 
	        derivative_estimator_values(q_time) = fmax(derivative_estimator_values(q_time), fabs(L2_projection_f_time_derivative_values[q_time][0] - solution_second_time_derivative_values(q_time) - (1/dt)*(1/dt)*Q_second_derivative_values(q_time)*(solution_values(q_space) - old_solution_values(q_space + (no_q_time - 1)*no_q_space)) + solution_time_derivative_laplacian_values(q_space + q_time*no_q_space) + (1/dt)*Q_derivative_values(q_time)*space_estimator_jump_value));
	        }
        }

        for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
        {
        space_estimator_values.block(q_time)(cell_no) = C_cell*estimator_values(q_time);
        space_derivative_estimator_values.block(q_time)(cell_no) = C_cell*derivative_estimator_values(q_time);
        }

    estimator_values = 0; derivative_estimator_values = 0;

        for (unsigned int face = 0; face < 2*dim; ++face)
        {
        if (space_cell->face(face)->at_boundary() == false && space_cell->face(face)->has_children() == false && space_cell->neighbor_is_coarser(face) == false)
        {
		typename DoFHandler<dim>::active_cell_iterator space_cell_neighbor = space_cell->neighbor (face);
		const unsigned int neighbor_face_no = space_cell->neighbor_face_no (face);
         
	    fe_values_space_face.reinit (space_cell, face); fe_values_space_face_neighbor.reinit (space_cell_neighbor, neighbor_face_no);
		const std::vector<Tensor<1,dim>> &normals = fe_values_space_face_neighbor.get_normal_vectors ();

		fe_values_space_face.get_function_gradients (old_solution_plus, solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (old_solution_plus, solution_face_gradient_neighbor_values);

		    for (unsigned int q_space = 0; q_space < no_q_space_face; ++q_space)
		    {
		    jump_values(q_space) = solution_face_gradient_neighbor_values[q_space]*normals[q_space] - solution_face_gradient_values[q_space]*normals[q_space];
		    }

		switch(time_degree)
		{
		case 0: fe_values_space_face.get_function_gradients (solution, solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (solution, solution_face_gradient_neighbor_values); break;
		default: fe_values_space_face.get_function_gradients (reordered_solution.block(0), solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (reordered_solution.block(0), solution_face_gradient_neighbor_values);
		}

	        for (unsigned int q_space = 0; q_space < no_q_space_face; ++q_space)
		    {
		    jump_values(q_space) += solution_face_gradient_values[q_space]*normals[q_space] - solution_face_gradient_neighbor_values[q_space]*normals[q_space];
		    }

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
		    {
		    if (time_degree > 0)
	     	{
	        if (q_time > 0) {fe_values_space_face.get_function_gradients (reordered_solution_at_temporal_quadrature_points.block(q_time), solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (reordered_solution_at_temporal_quadrature_points.block(q_time), solution_face_gradient_neighbor_values);}
	        fe_values_space_face.get_function_gradients (reordered_solution_time_derivative_at_temporal_quadrature_points.block(q_time), solution_time_derivative_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (reordered_solution_time_derivative_at_temporal_quadrature_points.block(q_time), solution_time_derivative_face_gradient_neighbor_values);
            }
			      
				for (unsigned int q_space = 0; q_space < no_q_space_face; ++q_space)
		        {			
		       	estimator_values(q_time) = fmax(estimator_values(q_time), fabs(solution_face_gradient_values[q_space]*normals[q_space] - solution_face_gradient_neighbor_values[q_space]*normals[q_space] + Q_values(q_time)*jump_values(q_space)));
			    derivative_estimator_values(q_time) = fmax(derivative_estimator_values(q_time), fabs(solution_time_derivative_face_gradient_values[q_space]*normals[q_space] - solution_time_derivative_face_gradient_neighbor_values[q_space]*normals[q_space] + (1/dt)*Q_derivative_values(q_time)*jump_values(q_space)));
			    }
            }	 
        }
        if (space_cell->face(face)->at_boundary() == false && space_cell->face(face)->has_children() == false && space_cell->neighbor_is_coarser(face) == true)
        {
        typename DoFHandler<dim>::active_cell_iterator space_cell_neighbor = space_cell->neighbor (face);
        std::pair<unsigned int, unsigned int> neighbor_face_no = space_cell->neighbor_of_coarser_neighbor (face);

	    fe_values_space_face.reinit (space_cell, face); fe_values_space_subface.reinit (space_cell_neighbor, neighbor_face_no.first, neighbor_face_no.second);
        const std::vector<Tensor<1,dim>> &normals = fe_values_space_subface.get_normal_vectors ();

		fe_values_space_face.get_function_gradients (old_solution_plus, solution_face_gradient_values); fe_values_space_subface.get_function_gradients (old_solution_plus, solution_face_gradient_neighbor_values);

            for (unsigned int q_space = 0; q_space < no_q_space_face; ++q_space)
		    {
		    jump_values(q_space) = solution_face_gradient_neighbor_values[q_space]*normals[q_space] - solution_face_gradient_values[q_space]*normals[q_space];
		    }

		switch(time_degree)
		{
		case 0: fe_values_space_face.get_function_gradients (solution, solution_face_gradient_values); fe_values_space_subface.get_function_gradients (solution, solution_face_gradient_neighbor_values); break;
		default: fe_values_space_face.get_function_gradients (reordered_solution.block(0), solution_face_gradient_values); fe_values_space_subface.get_function_gradients (reordered_solution.block(0), solution_face_gradient_neighbor_values);
		}

            for (unsigned int q_space = 0; q_space < no_q_space_face; ++q_space)
		    {
		    jump_values(q_space) += solution_face_gradient_values[q_space]*normals[q_space] - solution_face_gradient_neighbor_values[q_space]*normals[q_space];
		    }

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
		    {
		    if (time_degree > 0)
	     	{
	        if (q_time > 0) {fe_values_space_face.get_function_gradients (reordered_solution_at_temporal_quadrature_points.block(q_time), solution_face_gradient_values); fe_values_space_subface.get_function_gradients (reordered_solution_at_temporal_quadrature_points.block(q_time), solution_face_gradient_neighbor_values);}
	        fe_values_space_face.get_function_gradients (reordered_solution_time_derivative_at_temporal_quadrature_points.block(q_time), solution_time_derivative_face_gradient_values); fe_values_space_subface.get_function_gradients (reordered_solution_time_derivative_at_temporal_quadrature_points.block(q_time), solution_time_derivative_face_gradient_neighbor_values);
            }

                for (unsigned int q_space = 0; q_space < no_q_space_face; ++q_space)
		        {			
		       	estimator_values(q_time) = fmax(estimator_values(q_time), fabs(solution_face_gradient_values[q_space]*normals[q_space] - solution_face_gradient_neighbor_values[q_space]*normals[q_space] + Q_values(q_time)*jump_values(q_space)));
			    derivative_estimator_values(q_time) = fmax(derivative_estimator_values(q_time), fabs(solution_time_derivative_face_gradient_values[q_space]*normals[q_space] - solution_time_derivative_face_gradient_neighbor_values[q_space]*normals[q_space] + (1/dt)*Q_derivative_values(q_time)*jump_values(q_space)));
			    }
            }
        }
        if (space_cell->face(face)->at_boundary() == false && space_cell->face(face)->has_children() == true && space_cell->neighbor_is_coarser(face) == false)
        {
        const unsigned int no_of_subfaces = space_cell->face(face)->n_children();
        const unsigned int neighbor_face_no = space_cell->neighbor_of_neighbor (face);

            for (unsigned int subface = 0; subface < no_of_subfaces; ++subface)
            {
            typename DoFHandler<dim>::active_cell_iterator space_cell_neighbor = space_cell->neighbor_child_on_subface (face, subface);

            fe_values_space_subface.reinit (space_cell, face, subface); fe_values_space_face_neighbor.reinit (space_cell_neighbor, neighbor_face_no);
            const std::vector<Tensor<1,dim>> &normals = fe_values_space_subface.get_normal_vectors ();

            fe_values_space_subface.get_function_gradients (old_solution_plus, solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (old_solution_plus, solution_face_gradient_neighbor_values);

		        for (unsigned int q_space = 0; q_space < no_q_space_face; ++q_space)
		        {
		        jump_values(q_space) = solution_face_gradient_neighbor_values[q_space]*normals[q_space] - solution_face_gradient_values[q_space]*normals[q_space];
		        }

	       	switch(time_degree)
		    {
		    case 0: fe_values_space_subface.get_function_gradients (solution, solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (solution, solution_face_gradient_neighbor_values); break;
		    default: fe_values_space_subface.get_function_gradients (reordered_solution.block(0), solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (reordered_solution.block(0), solution_face_gradient_neighbor_values);
		    }

	            for (unsigned int q_space = 0; q_space < no_q_space_face; ++q_space)
		        {
		        jump_values(q_space) += solution_face_gradient_values[q_space]*normals[q_space] - solution_face_gradient_neighbor_values[q_space]*normals[q_space];
		        }

                for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
		        {
		        if (time_degree > 0)
	         	{
	            if (q_time > 0) {fe_values_space_subface.get_function_gradients (reordered_solution_at_temporal_quadrature_points.block(q_time), solution_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (reordered_solution_at_temporal_quadrature_points.block(q_time), solution_face_gradient_neighbor_values);}
	            fe_values_space_subface.get_function_gradients (reordered_solution_time_derivative_at_temporal_quadrature_points.block(q_time), solution_time_derivative_face_gradient_values); fe_values_space_face_neighbor.get_function_gradients (reordered_solution_time_derivative_at_temporal_quadrature_points.block(q_time), solution_time_derivative_face_gradient_neighbor_values);
                }
			      
				    for (unsigned int q_space = 0; q_space < no_q_space_face; ++q_space)
		            {			
		       	    estimator_values(q_time) = fmax(estimator_values(q_time), fabs(solution_face_gradient_values[q_space]*normals[q_space] - solution_face_gradient_neighbor_values[q_space]*normals[q_space] + Q_values(q_time)*jump_values(q_space)));
			        derivative_estimator_values(q_time) = fmax(derivative_estimator_values(q_time), fabs(solution_time_derivative_face_gradient_values[q_space]*normals[q_space] - solution_time_derivative_face_gradient_neighbor_values[q_space]*normals[q_space] + (1/dt)*Q_derivative_values(q_time)*jump_values(q_space)));
			        }
                }	
            }
        }
        }

        for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
        {
        space_estimator_values.block(q_time)(cell_no) += C_edge*estimator_values(q_time);
        space_derivative_estimator_values.block(q_time)(cell_no) += C_edge*derivative_estimator_values(q_time);
        }
    }

    Vector<double> reconstructed_solution_at_q_time (no_of_space_dofs);

        for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
        {
        switch(time_degree)
        {
        case 0: reconstructed_solution_at_q_time = solution; reconstructed_solution_at_q_time *= 1 + Q_values(q_time); reconstructed_solution_at_q_time.add(-Q_values(q_time), old_solution_plus); break;
        default: reconstructed_solution_at_q_time = reordered_solution_at_temporal_quadrature_points.block(q_time); reconstructed_solution_at_q_time.add(Q_values(q_time), reordered_solution_at_temporal_quadrature_points.block(0)); reconstructed_solution_at_q_time.add(-Q_values(q_time), old_solution_plus);
        }

        double space_estimator_at_q_time = space_estimator_values.block(q_time).linfty_norm();

            for (unsigned int cell_no = 0; cell_no < no_of_cells; ++cell_no)
            {
            refinement_vector(cell_no) += (space_estimator_values.block(q_time)(cell_no)*(2*reconstructed_solution_at_q_time.linfty_norm() + space_estimator_at_q_time) + space_derivative_estimator_values.block(q_time)(cell_no))*fe_values_time.JxW(q_time);
            }

        etaS += (space_estimator_at_q_time*(2*reconstructed_solution_at_q_time.linfty_norm() + space_estimator_at_q_time) + space_derivative_estimator_values.block(q_time).linfty_norm())*fe_values_time.JxW(q_time);
        }

    refinement_vector *= 1/dt;
}

template <int dim> void dGcGblowup<dim>::compute_time_estimator (const unsigned int &no_q_space_x, const unsigned int &no_q_time)
{
const QGaussLobatto<dim> quadrature_formula_space (no_q_space_x); const QGaussLobatto<1> quadrature_formula_time (no_q_time);

FEValues<dim> fe_values_space (fe_space, quadrature_formula_space, update_values | update_quadrature_points);
FEValues<1> fe_values_time (fe_time, quadrature_formula_time, update_values | update_gradients | update_quadrature_points | update_JxW_values);
FEValues<1> old_fe_values_time (old_fe_time, quadrature_formula_time, update_values | update_gradients | update_JxW_values);

const unsigned int no_q_space = quadrature_formula_space.size();
const unsigned int dofs_per_cell = fe.dofs_per_cell;

FullMatrix<double> temporal_mass_matrix_inv (time_degree + 1, time_degree + 1);
FullMatrix<double> old_temporal_mass_matrix_inv (time_degree + 1, time_degree + 1);

Vector<double> estimator_values (no_q_time);
Vector<double> solution_values (no_q_space*no_q_time);
Vector<double> old_solution_values (no_q_space*no_q_time);
Vector<double> Q_values (no_q_time);
Vector<double> Q_derivative_values (no_q_time);
Vector<double> L2_projection_rhs (time_degree + 1);
Vector<double> L2_projection_f (time_degree + 1);
std::vector<double> L2_projection_f_values (no_q_time);
std::vector<double> old_old_solution_plus_values (no_q_space);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

if (time_degree > 0)
{
create_temporal_mass_matrix (fe_time, temporal_mass_matrix_inv); temporal_mass_matrix_inv.gauss_jordan ();

if (dt == dt_old) {old_temporal_mass_matrix_inv = temporal_mass_matrix_inv;}
else {create_temporal_mass_matrix (old_fe_time, old_temporal_mass_matrix_inv); old_temporal_mass_matrix_inv.gauss_jordan ();}
}

typename DoFHandler<1>::active_cell_iterator time_cell = dof_handler_time.begin_active(); typename DoFHandler<1>::active_cell_iterator old_time_cell = old_dof_handler_time.begin_active();
fe_values_time.reinit (time_cell); old_fe_values_time.reinit (old_time_cell);

    for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
    {
    compute_Q_values (time_degree, (2/dt)*fe_values_time.quadrature_point(q_time)(0) - 1, Q_values(q_time), Q_derivative_values(q_time), etaT);
    }

typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active (), final_cell = dof_handler.end ();
typename DoFHandler<dim>::active_cell_iterator space_cell = dof_handler_space.begin_active ();

etaT = 0; double discrete_laplacian_jump_value = 0; double nonlinearity_value = 0; double solution_time_derivative_value = 0;

    for (; cell != final_cell; ++cell, ++space_cell)
    {
    fe_values_space.reinit (space_cell);

    cell->get_dof_indices (local_dof_indices);

    get_spacetime_function_values (solution, fe_values_space, fe_values_time, local_dof_indices, solution_values);

    if (timestep_number > 1) {get_spacetime_function_values (old_solution, fe_values_space, old_fe_values_time, local_dof_indices, old_solution_values); fe_values_space.get_function_values (old_old_solution_plus, old_old_solution_plus_values);}
    else {fe_values_space.get_function_values (old_solution_plus, old_old_solution_plus_values); for (unsigned int q_space = 0; q_space < no_q_space; ++q_space) {old_solution_values (q_space + (no_q_time - 1)*no_q_space) = old_old_solution_plus_values[q_space];} initialvalueslaplacian<dim>().value_list (fe_values_space.get_quadrature_points(), old_old_solution_plus_values);}

        for (unsigned int q_space = 0; q_space < no_q_space; ++q_space)
        {
        if (timestep_number > 1)
        {
        switch (time_degree) {case 0: L2_projection_f(time_degree) = old_solution_values(q_space)*old_solution_values(q_space); break;
        default: L2_projection_f(time_degree) = 0; L2_projection_rhs = 0; solution_time_derivative_value = 0;

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            nonlinearity_value = old_solution_values(q_space+q_time*no_q_space)*old_solution_values(q_space+q_time*no_q_space)*old_fe_values_time.JxW(q_time);

	            for (unsigned int i = 0; i < time_degree + 1; ++i)
	            {
	            L2_projection_rhs(i) += nonlinearity_value*old_fe_values_time.shape_value(i, q_time);
	            }
            }

            for (unsigned int i = 0; i < time_degree + 1; ++i)
            {
            L2_projection_f(time_degree) += old_temporal_mass_matrix_inv(time_degree, i)*L2_projection_rhs(i);
            }

	        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
            const unsigned int comp_s_i = fe.system_to_component_index(i).second; const unsigned int comp_t_i = fe.system_to_component_index(i).first;

            solution_time_derivative_value += old_solution(local_dof_indices[i])*fe_values_space.shape_value(comp_s_i, q_space)*old_fe_values_time.shape_grad(comp_t_i, no_q_time - 1)[0];
            }
        }

        discrete_laplacian_jump_value = -L2_projection_f(time_degree) + solution_time_derivative_value + (1/dt_old)*Q_derivative_values(no_q_time-1)*(old_solution_values(q_space) - old_old_solution_plus_values[q_space]);
        }
        else
        {
        discrete_laplacian_jump_value = a*old_old_solution_plus_values[q_space];
        }

        switch (time_degree) {case 0: for (unsigned int q_time = 0; q_time < no_q_time; ++q_time) {L2_projection_f_values[q_time] = solution_values(q_space)*solution_values(q_space);} break;
        default: L2_projection_rhs = 0; solution_time_derivative_value = 0;

            for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            nonlinearity_value = solution_values(q_space+q_time*no_q_space)*solution_values(q_space+q_time*no_q_space)*fe_values_time.JxW(q_time);

                for (unsigned int i = 0; i < time_degree + 1; ++i)
                {
                L2_projection_rhs(i) += nonlinearity_value*fe_values_time.shape_value(i, q_time);
     	        }
            }

        temporal_mass_matrix_inv.vmult (L2_projection_f, L2_projection_rhs);
        fe_values_time.get_function_values (L2_projection_f, L2_projection_f_values);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
            const unsigned int comp_s_i = fe.system_to_component_index(i).second; const unsigned int comp_t_i = fe.system_to_component_index(i).first;

            solution_time_derivative_value += solution(local_dof_indices[i])*fe_values_space.shape_value(comp_s_i, q_space)*fe_values_time.shape_grad(comp_t_i, 0)[0];
            }
        }

        discrete_laplacian_jump_value += L2_projection_f_values[0] - solution_time_derivative_value - (1/dt)*Q_derivative_values(0)*(solution_values(q_space) - old_solution_values(q_space+(no_q_time-1)*no_q_space));

	        for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
            {
            estimator_values(q_time) = fmax(estimator_values(q_time), fabs((solution_values(q_space + q_time*no_q_space) + Q_values(q_time)*(solution_values(q_space) - old_solution_values(q_space + (no_q_time - 1)*no_q_space)))*(solution_values(q_space + q_time*no_q_space) + Q_values(q_time)*(solution_values(q_space) - old_solution_values(q_space + (no_q_time - 1)*no_q_space))) - L2_projection_f_values[q_time] - Q_values(q_time)*discrete_laplacian_jump_value));
            }
        }
    }

    for (unsigned int q_time = 0; q_time < no_q_time; ++q_time)
    {
    etaT += estimator_values(q_time)*fe_values_time.JxW(q_time);
    }
}

template <int dim>
void dGcGblowup<dim>::compute_estimator ()
{
Vector<double> reconstructed_solution_at_quadrature_point (dof_handler_space.n_dofs());

switch(time_degree)
{
case 0: reconstructed_solution_at_quadrature_point = solution; reconstructed_solution_at_quadrature_point *= 0.5; reconstructed_solution_at_quadrature_point.add(0.5, old_solution_plus); solution_time_integral = dt*reconstructed_solution_at_quadrature_point.linfty_norm(); break;
default:
const QGaussLobatto<1> quadrature_formula_time (time_degree + 1);
FEValues<1> fe_values_time (fe_time, quadrature_formula_time, update_quadrature_points | update_JxW_values);
typename DoFHandler<1>::active_cell_iterator time_cell = dof_handler_time.begin_active();
fe_values_time.reinit (time_cell);

solution_time_integral = 0; double Q_value = 0; double Q_derivative_value = 0;

for (unsigned int q_time = 0; q_time < time_degree + 1; ++q_time)
{
reconstructed_solution_at_quadrature_point = reordered_solution.block(q_time);
compute_Q_values (time_degree, (2/dt)*fe_values_time.quadrature_point(q_time)(0) - 1, Q_value, Q_derivative_value, Q_derivative_value);
reconstructed_solution_at_quadrature_point.add(Q_value, reordered_solution.block(0)); reconstructed_solution_at_quadrature_point.add(-Q_value, old_solution_plus);

solution_time_integral += reconstructed_solution_at_quadrature_point.linfty_norm()*fe_values_time.JxW(q_time);
}
}

estimator = estimator + etaS + etaT;
delta = delta + 0.05;

delta_residual = 1 + delta*(2*solution_time_integral - 1) + 2*dt*estimator*delta*delta;

for (unsigned int i = 0; i < 10; ++i)
{
delta = delta - delta_residual/(2*solution_time_integral - 1 + 4*dt*estimator*delta);
delta_residual = 1 + delta*(2*solution_time_integral - 1) + 2*dt*estimator*delta*delta;
if (fabs(delta_residual) < 1e-15) {break;}
}

r = exp(2*solution_time_integral + dt*delta*estimator);
estimator = r*estimator;

if (fabs(delta_residual) > delta_residual_threshold)
{
deallog << std::endl << "No solution to the delta equation found -- aborting!" << std::endl;
}
else
{
deallog << std::endl << "max||U(t)||: " << solution.linfty_norm() << std::endl; // Output a (crude) approximation to the LinftyLinfty norm of the numerical solution
deallog << "Estimator: " << estimator << std::endl; // Output the value of the estimator
deallog << "Space Estimator: " << etaS << std::endl; // Output the value of the time estimator
deallog << "Time Estimator: " << etaT << std::endl; // Output the value of the time estimator
deallog << "r: " << r << std::endl << std::endl; // Output the value of the scaling parameter r_m
}
}

template <int dim>
void dGcGblowup<dim>::run ()
{
deallog << "Spatial Polynomial Degree: " << space_degree << std::endl;
deallog << "Temporal Polynomial Degree: " << time_degree << std::endl;

// Refine the mesh based on the initial condition

deallog << std::endl << "Refining the spatial mesh based on the initial condition..." << std::endl;

GridGenerator::hyper_cube (triangulation_space, -5, 5); triangulation_space.refine_global (2);
//GridGenerator::hyper_cube (triangulation_space, -9, 9); triangulation_space.refine_global (2);

refine_initial_mesh ();

// Setup meshes
old_triangulation_space.copy_triangulation (triangulation_space); old_old_triangulation_space.copy_triangulation (triangulation_space);
GridGenerator::hyper_cube (triangulation_time, 0, dt); old_triangulation_time.copy_triangulation (triangulation_time);

deallog << std::endl << "Setting up the initial mesh and time step length on the first time step..." << std::endl;

    for (; fabs(delta_residual) < delta_residual_threshold; ++timestep_number)
    {
    if (timestep_number == 0)
    {
    while (mesh_change == true)
    {
    mesh_change = false;

    setup_system_full ();

    deallog << std::endl << "Spatial Degrees of Freedom: " << dof_handler_space.n_dofs() << std::endl;
    deallog << "\u0394t: " << dt << std::endl;
    deallog << "Projecting the initial condition..." << std::endl;

    energy_project (2*space_degree + 1, initialvalueslaplacian<dim>(), solution_plus); old_solution_plus = solution_plus;
    output_solution ();
    assemble_and_solve (int((3*space_degree + 1)/2) + 1, int((3*time_degree + 1)/2) + 1, 20, 1e-8); // Setup and solve the system and output the numerical solution
    compute_space_estimator (int((3*space_degree + 3)/2) + 1, int((3*time_degree + 3)/2) + 1); // Compute the space estimator
    compute_time_estimator (int((3*space_degree + 3)/2) + 1, int((3*time_degree + 3)/2) + 1); // Compute the time estimator

    deallog << "Space Estimator: " << etaS << std::endl; // Output the value of the time estimator
    deallog << "Time Estimator: " << etaT << std::endl; // Output the value of the time estimator

    refine_mesh ();

    if (etaT > temporal_refinement_threshold)
    {
    dt = 0.5*dt; triangulation_time.clear(); GridGenerator::hyper_cube (triangulation_time, 0, dt); old_triangulation_time.clear(); old_triangulation_time.copy_triangulation (triangulation_time);
    mesh_change = true;
    }
    }
    }
    else
    {
    assemble_and_solve (int((3*space_degree + 1)/2) + 1, int((3*time_degree + 1)/2) + 1, 20, 1e-8); // Setup and solve the system and output the numerical solution
    compute_time_estimator (int((3*space_degree + 3)/2) + 1, int((3*time_degree + 3)/2) + 1); // Compute the time estimator

    if (etaT > temporal_refinement_threshold)
    {
    dt = 0.5*dt; triangulation_time.clear(); GridGenerator::hyper_cube (triangulation_time, 0, dt);

    setup_system_time ();
    assemble_and_solve (int((3*space_degree + 1)/2) + 1, int((3*time_degree + 1)/2) + 1, 20, 1e-8); // Setup and solve the system and output the numerical solution
    compute_time_estimator (int((3*space_degree + 3)/2) + 1, int((3*time_degree + 3)/2) + 1); // Compute the time estimator
    }

    compute_space_estimator (int((3*space_degree + 3)/2) + 1, int((3*time_degree + 3)/2) + 1); // Compute the space estimator
    }

    if (timestep_number == 0) {timestep_number = 1;}

    time = time + dt;

    deallog  << std::endl << "Time step " << timestep_number << " at t=" << time << std::endl;
    deallog << "Total Degrees of Freedom: " << dof_handler.n_dofs () << std::endl;
    deallog << "Spatial Degrees of Freedom: " << dof_handler_space.n_dofs () << std::endl;
    deallog << "\u0394t: " << dt << std::endl;

    output_solution ();
    compute_estimator ();

    old_solution = solution; old_old_solution_plus = old_solution_plus; old_solution_plus = solution_plus; 
    spatial_refinement_threshold *= r; spatial_coarsening_threshold *= r; temporal_refinement_threshold *= r;
    mesh_change = false;

    if (dt != dt_old)
    {
    dt_old = dt; old_triangulation_time.clear(); old_triangulation_time.copy_triangulation(triangulation_time); old_dof_handler_time.distribute_dofs (old_fe_time);
    }

    }
}

int main ()
{
deallog.depth_console (2);
std::ofstream logfile ("deallog");
deallog.attach (logfile);

try
{
dGcGblowup<2> dGcG;
dGcG.run ();
}
catch (std::exception &exc)
{
std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
std::cerr << "Exception on processing: " << std::endl << exc.what() << std::endl << "Aborting!" << std::endl << "----------------------------------------------------" << std::endl;
return 1;
}
catch (...)
{
std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl << "----------------------------------------------------" << std::endl;
return 1;
};

return 0;
}