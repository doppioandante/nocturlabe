#pragma once

#include <vector>
#include <functional>

#include <ida/ida.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>

#include <cppad/cppad.hpp>

namespace nocturlabe 
{

using CppAD::AD;

template <typename DAEParams, typename Scalar = realtype>
class IDASolver
{
public:

    template <typename T>
    using ODEFunction = std::function<
        std::vector<T>(
            const DAEParams* params,
            const std::vector<T>&,
            const std::vector<T>&,
            const std::vector<T>&
        )>;

    template <typename T>
    using ALGFunction = std::function<
        std::vector<T>(
            const DAEParams* params,
            const std::vector<T>&,
            const std::vector<T>&,
            const std::vector<T>&
        )>;

private:
    struct IDAUserData {
        CppAD::ADFun<Scalar> F;
        ODEFunction<Scalar> f;
        ALGFunction<Scalar> g;
        const DAEParams* params;
        std::vector<Scalar>* inputs;
        int n;
        int m;
        int p;
        int q;
    };

public:
    IDASolver(size_t num_diffeq, size_t num_algeq, size_t num_algvar, size_t num_inputs,
                  ODEFunction<Scalar> f, ALGFunction<Scalar> g,
                  ODEFunction<AD<Scalar>> f_ad, ALGFunction<AD<Scalar>> g_ad);
    ~IDASolver();

    void set_params(const DAEParams* params);
    void set_avtol(const std::vector<realtype> avtol);
    void set_rtol(realtype rtol);

    void set_initial_state(const std::vector<realtype>& x0, const std::vector<realtype>& u0);
    void start(realtype t0, realtype first_step, bool fix_initial_conditions = true);

    void set_input(const std::vector<realtype>& u0);
    int do_step(realtype step, realtype& tret);

    void get_state(std::vector<realtype>& x, std::vector<realtype>& z) const;

private:
    static int fres(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void* user_data);
    static int jacfresCSR(realtype tt,  realtype cj,
                          N_Vector yy, N_Vector yp, N_Vector resvec,
                          SUNMatrix JJ, void *user_data,
                          N_Vector tempv1, N_Vector tempv2, N_Vector tempv3);

private:
    size_t num_diffeq_, num_algeq_, num_algvar_, num_inputs_, N_, M_;
    realtype rtol_;
    const DAEParams* params_;
    std::vector<realtype> u;

    IDAUserData UserData_;

    ODEFunction<Scalar> f_;
    ALGFunction<Scalar> g_;
    ODEFunction<AD<Scalar>> f_sym;
    ALGFunction<AD<Scalar>> g_sym;

    SUNContext ctx_;
    void* mem_;
    N_Vector yy_, yp_, avtol_;
    SUNMatrix ls_matrix_;
    SUNLinearSolver LS_;

    realtype t_;
};

template <typename DAEParams, typename Scalar>
IDASolver<DAEParams, Scalar>::IDASolver(
    size_t num_diffeq, size_t num_algeq, size_t num_algvar, size_t num_inputs,
    ODEFunction<Scalar> f, ALGFunction<Scalar> g,
    ODEFunction<AD<Scalar>> f_ad, ODEFunction<AD<Scalar>> g_ad
)
    : num_diffeq_(num_diffeq), num_algeq_(num_algeq), num_algvar_(num_algvar), num_inputs_(num_inputs),
      N_(num_algvar + num_diffeq), M_(num_algeq + num_diffeq), rtol_(RCONST(1e-6)), params_(nullptr),
      f_(f), g_(g),
      f_sym(f_ad), g_sym(g_ad)
{
    SUNContext_Create(NULL, &ctx_);
    mem_ = IDACreate(ctx_);

    yy_ = N_VNew_Serial(N_, ctx_);
    yp_ = N_VClone(yy_);
    avtol_ = N_VClone(yy_);

    ls_matrix_ = SUNSparseMatrix(M_, N_, M_ * N_, CSR_MAT, ctx_);
    LS_ = SUNLinSol_KLU(yy_, ls_matrix_, ctx_);
}

template <typename DAEParams, typename Scalar>
IDASolver<DAEParams, Scalar>::~IDASolver() {
    if (yy_ != nullptr) {
        N_VDestroy(yy_);
    }
    if (yp_ != nullptr) {
        N_VDestroy(yp_);
    }
    if (avtol_ != nullptr) {
        N_VDestroy(avtol_);
    }
    if (ls_matrix_ != nullptr) {
        SUNMatDestroy(ls_matrix_);
    }
    if (LS_ != nullptr) {
        SUNLinSolFree(LS_);
    }
    if (mem_ != nullptr) {
        IDAFree(&mem_);
    }
    if (ctx_ != nullptr) {
        SUNContext_Free(&ctx_);
    }
}


template <typename DAEParams, typename Scalar>
void IDASolver<DAEParams, Scalar>::set_params(const DAEParams* params)
{
    params_ = params;
}

template <typename DAEParams, typename Scalar>
void IDASolver<DAEParams, Scalar>::set_avtol(const std::vector<realtype> avtol)
{
    realtype* vptr = N_VGetArrayPointer(avtol_);
    for (size_t i = 0; i < N_; i++) {
        vptr[i] = avtol.at(i);
    }
}

template <typename DAEParams, typename Scalar>
void IDASolver<DAEParams, Scalar>::set_rtol(realtype rtol)
{
    rtol_ = rtol;
}

template <typename DAEParams, typename Scalar>
void IDASolver<DAEParams, Scalar>::set_initial_state(const std::vector<realtype>& x0, const std::vector<realtype>& u0)
{
    u = u0;

    realtype* yptr = N_VGetArrayPointer(yy_);

    for (size_t i = 0; i < N_; i++) {
        yptr[i] = x0.at(i);
    }

    std::vector<realtype> y_0(x0.begin(), x0.begin() + num_diffeq_);
    std::vector<realtype> z_0(x0.begin() + num_diffeq_, x0.end());

    auto yp_0 = f_(params_, y_0, z_0, u0);
    yp_0.insert(std::end(yp_0), std::cbegin(z_0), std::cend(z_0));
    yptr = N_VGetArrayPointer(yp_);
    std::copy(std::cbegin(yp_0), std::cend(yp_0), yptr);
}

template <typename DAEParams, typename Scalar>
void IDASolver<DAEParams, Scalar>::start(realtype t0, realtype first_step, bool fix_initial_conditions)
{
    t_ = t0;
    IDAInit(mem_, &IDASolver<DAEParams, Scalar>::fres, t0, yy_, yp_);
    IDASVtolerances(mem_, rtol_, avtol_);

    std::vector< AD<realtype> > X(N_ + num_inputs_);
    CppAD::Independent(X);

    auto x = std::vector< AD<realtype> >(X.begin(), X.begin() + num_diffeq_);
    auto z = std::vector< AD<realtype> >(X.begin() + num_diffeq_, X.begin() + N_);
    auto u = std::vector< AD<realtype> >(X.begin() + N_, X.end());
    auto y = f_sym(params_, x, z, u);
    auto z_res = g_sym(params_, x, z, u);
    auto Y = std::vector< AD<realtype> >(y);
    Y.insert(std::end(Y), z_res.cbegin(), z_res.cend());
    UserData_.F = CppAD::ADFun<realtype>{X, Y};
    UserData_.F.optimize();
    UserData_.f = f_;
    UserData_.g = g_;
    UserData_.params = params_;
    UserData_.inputs = &this->u;
    UserData_.n = num_diffeq_;
    UserData_.m = num_algeq_;
    UserData_.p = num_algvar_;
    UserData_.q = num_inputs_;

    IDASetUserData(mem_, &UserData_);
    IDASetLinearSolver(mem_, LS_, ls_matrix_);
    IDASetJacFn(mem_, &IDASolver<DAEParams, Scalar>::jacfresCSR);

    if (fix_initial_conditions) {
        auto ic_vars = N_VClone(yy_);
        realtype* vptr = N_VGetArrayPointer(ic_vars);
        for (size_t i = 0; i < N_; i++) {
            if (i < num_diffeq_) {
                vptr[i] = RCONST(1.0);
            } else {
                vptr[i] = RCONST(0.0);
            }
        }
        IDASetId(mem_, ic_vars);
        N_VDestroy(ic_vars);
        IDACalcIC(mem_, IDA_YA_YDP_INIT, first_step);
    }
}

template <typename DAEParams, typename Scalar>
int IDASolver<DAEParams, Scalar>::do_step(realtype step, realtype& tret)
{
    int retval = IDASolve(mem_, t_ + step, &tret, yy_, yp_, IDA_NORMAL);
    if (retval == IDA_SUCCESS || retval == IDA_TSTOP_RETURN) {
        IDAReInit(mem_, tret, yy_, yp_);
        t_ = tret;
    }
    return retval;
}

template <typename DAEParams, typename Scalar>
void IDASolver<DAEParams, Scalar>::get_state(std::vector<realtype>& x, std::vector<realtype>& z) const
{
    realtype* yptr = N_VGetArrayPointer(yy_);

    x.assign(yptr, yptr + num_diffeq_);
    z.assign(yptr + num_diffeq_, yptr + N_);
}

template <typename DAEParams, typename Scalar>
int IDASolver<DAEParams, Scalar>::fres(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void* user_data)
{
    auto udata = (IDAUserData*)user_data;
    Scalar *yval, *ypval, *rval;

    yval = N_VGetArrayPointer(yy);
    ypval = N_VGetArrayPointer(yp);
    rval = N_VGetArrayPointer(rr);

    std::vector<Scalar> x{yval, yval + udata->n};
    std::vector<Scalar> z{yval + udata->n, yval + (udata->n + udata->p)};
    std::vector<Scalar> fval = udata->f(udata->params, x, z, *udata->inputs);
    std::vector<Scalar> g_res = udata->g(udata->params, x, z, *udata->inputs);

    for (int i = 0; i < udata->n; i++) {
        rval[i] = ypval[i] - fval[i];
    }

    for (int i = 0; i < udata->m; i++) {
        rval[udata->n + i] = g_res[i];
    }

    return 0;
}

/*
  tt – is the current value of the independent variable t
  cj – is the scalar in the system Jacobian, proportional to the inverse of the step size (alpha in (5.6)). 
  yy – is the current value of the dependent variable vector, y(t)
  yp – is the current value of ydot(t)
  rr – is the current value of the residual vector F(t, y, ydot)
  Jac – is the output (approximate) Jacobian matrix (of type SUNMatrix), J=dF/dy + cj dF/Dydot
  user_data - is a pointer to user data, the same as the user_data parameter passed to IDASetUserData().
  tmp1, tmp2, and tmp3 – are pointers to memory allocated for variables of type N_Vector which can be used by IDALsJacFn function as temporary storage or work space.
*/
template <typename DAEParams, typename Scalar>
int IDASolver<DAEParams, Scalar>::jacfresCSR(realtype tt,  realtype cj,
            N_Vector yy, N_Vector yp, N_Vector resvec,
            SUNMatrix JJ, void *user_data,
            N_Vector tempv1, N_Vector tempv2, N_Vector tempv3)
{
    auto u = (IDAUserData*)user_data;

    realtype *yval;
    yval = N_VGetArrayPointer(yy);

    sunindextype *rowptrs = SUNSparseMatrix_IndexPointers(JJ);
    sunindextype *colvals = SUNSparseMatrix_IndexValues(JJ);
    realtype *data = SUNSparseMatrix_Data(JJ);

    SUNMatZero(JJ);

    int N = u->n + u->p + u->q;
    int M = u->n + u->m;
    std::vector<realtype> X(yval, yval + (u->n + u->p));
    X.insert(X.end(), u->inputs->begin(), u->inputs->end()); // evaluate Jacobian using current inputs
    // augment
    auto jac = u->F.Jacobian(X);

    int JacN = N - u->q;
    for (int i = 0; i <= M; i++) {
        rowptrs[i] = i * JacN;
    }

    // jacobian of f
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < JacN; j++) {
            data[JacN * i + j] = jac[N * i + j];
            if (i < u->n and i == j) {
                data[JacN * i + j] += cj;
            }
            colvals[JacN * i + j] = j;
        }
    }

    //static bool first = true;
    //if (first) {
    //    for (size_t i = 0; i < jac.size(); i++) {
    //        if ((i % N) == 0 and i > 0) std::cout << std::endl;
    //        std::cout << jac[i] << " ";
    //    }
    //    std::cout << std::endl;
    //    SUNSparseMatrix_Print(JJ, stdout);
    //    first = false;
    //}

    return 0;
}

}
