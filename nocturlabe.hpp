#pragma once

#include <vector>
#include <functional>
#include <exception>

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

class ida_error: public std::runtime_error
{
public:
    ida_error(const std::string& error, int retval) noexcept :
       std::runtime_error("ida error: " + error + " (retval: " + std::to_string(retval) + ")")
    {}
};


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
    inline static void check_result(int ida_res, const char* err) {
        if (ida_res < 0) {
            throw ida_error(err, ida_res);
        }
    }

private:
    struct IDAUserData {
        IDAUserData(IDAUserData&&) noexcept = default;
        IDAUserData(const IDAUserData&) = delete;
        IDAUserData() noexcept = default;

        CppAD::ADFun<Scalar> F;
        ODEFunction<Scalar> f;
        ALGFunction<Scalar> g;
        const DAEParams* params;
        std::vector<Scalar>* inputs;
        size_t n;
        size_t m;
        size_t p;
        size_t q;
    };

public:
    IDASolver(size_t num_diffeq, size_t num_algeq, size_t num_algvar, size_t num_inputs,
                  ODEFunction<Scalar> f, ALGFunction<Scalar> g,
                  ODEFunction<AD<Scalar>> f_ad, ALGFunction<AD<Scalar>> g_ad);
    ~IDASolver();
    IDASolver(const IDASolver&) = delete;
    IDASolver(IDASolver&&) noexcept;

    void set_params(const DAEParams* params);
    void set_avtol(const std::vector<realtype> avtol);
    void set_rtol(realtype rtol);

    void set_initial_state(const std::vector<realtype>& x0, const std::vector<realtype>& u0);
    void start(realtype t0, realtype first_step, bool fix_initial_conditions = true);

    void set_input(const std::vector<realtype>& u0);
    int do_step(realtype step, realtype& tret);
    int step_until(realtype t, realtype& tret);

    void get_state(std::vector<realtype>& x, std::vector<realtype>& z) const;
    void print_jacobian() const;

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

    std::unique_ptr<IDAUserData> UserData_;

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
    check_result(SUNContext_Create(NULL, &ctx_), "SUNContext_Create");
    mem_ = IDACreate(ctx_);
    if (mem_ == nullptr) {
        throw ida_error("IDACreate", 0);
    }

    yy_ = N_VNew_Serial(N_, ctx_);
    yp_ = N_VClone(yy_);
    avtol_ = N_VClone(yy_);

    ls_matrix_ = SUNSparseMatrix(M_, N_, M_ * N_, CSR_MAT, ctx_);
    LS_ = SUNLinSol_KLU(yy_, ls_matrix_, ctx_);
}

template <typename DAEParams, typename Scalar>
IDASolver<DAEParams, Scalar>::IDASolver(IDASolver&& rref) noexcept
    : num_diffeq_(rref.num_diffeq_),
      num_algeq_(rref.num_algeq_),
      num_algvar_(rref.num_algvar_),
      num_inputs_(rref.num_inputs_),
      N_(num_algvar_ + num_diffeq_),
      M_(num_algeq_ + num_diffeq_),
      rtol_(rref.rtol_),
      params_(rref.params_),
      f_(rref.f_),
      g_(rref.g_),
      f_sym(rref.f_sym),
      g_sym(rref.g_sym),
      u(std::move(rref.u)),
      UserData_(std::move(rref.UserData_)),
      ctx_(rref.ctx_),
      mem_(rref.mem_),
      avtol_(rref.avtol_),
      ls_matrix_(rref.ls_matrix_),
      LS_(rref.LS_),
      t_(rref.t_),
      yy_(rref.yy_),
      yp_(rref.yp_)
{
    rref.ctx_ = nullptr;
    rref.mem_ = nullptr;
    rref.LS_ = nullptr;
    rref.ls_matrix_ = nullptr;
    rref.avtol_ = nullptr;
    rref.yp_ = nullptr;
    rref.yy_ = nullptr;

    if (UserData_.get() != nullptr) {
        UserData_->inputs = &u;
    }
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
void IDASolver<DAEParams, Scalar>::set_input(const std::vector<realtype>& u0)
{
    u = u0;
}

template <typename DAEParams, typename Scalar>
void IDASolver<DAEParams, Scalar>::start(realtype t0, realtype first_step, bool fix_initial_conditions)
{
    t_ = t0;
    IDAInit(mem_, &IDASolver<DAEParams, Scalar>::fres, t0, yy_, yp_);
    IDASVtolerances(mem_, rtol_, avtol_);

    std::vector< AD<realtype> > X(N_ + num_inputs_);
    // set variables to the initial conditions so that AdCpp has more sensible values
    // for computing derivatives (in particular, it avoids division by zero)
    realtype* yptr = N_VGetArrayPointer(yy_);
    for (size_t i = 0; i < N_; i++) {
        X[i] = yptr[i];
    }
    for (size_t i = 0; i < num_inputs_; i++) {
        X[N_ + i] = u[i];
    }
    CppAD::Independent(X);

    auto x = std::vector< AD<realtype> >(X.begin(), X.begin() + num_diffeq_);
    auto z = std::vector< AD<realtype> >(X.begin() + num_diffeq_, X.begin() + N_);
    auto u = std::vector< AD<realtype> >(X.begin() + N_, X.end());
    auto y = f_sym(params_, x, z, u);
    auto z_res = g_sym(params_, x, z, u);
    auto Y = std::vector< AD<realtype> >(y);
    Y.insert(std::end(Y), z_res.cbegin(), z_res.cend());

    IDAUserData user_data = {
        .F = CppAD::ADFun<realtype>{X, Y},
        .f = f_,
        .g = g_,
        .params = params_,
        .inputs = &this->u,
        .n = num_diffeq_,
        .m = num_algeq_,
        .p = num_algvar_,
        .q = num_inputs_,
    };
    user_data.F.optimize();

    UserData_ = std::make_unique<IDAUserData>(std::move(user_data));

    IDASetUserData(mem_, UserData_.get());
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
    return step_until(t_ + step, tret);
}

template <typename DAEParams, typename Scalar>
int IDASolver<DAEParams, Scalar>::step_until(realtype t, realtype& tret)
{
    int retval = IDASolve(mem_, t, &tret, yy_, yp_, IDA_NORMAL);
    if (retval == IDA_SUCCESS || retval == IDA_TSTOP_RETURN) {
        //IDAReInit(mem_, tret, yy_, yp_);
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

//template <typename DAEParams, typename Scalar>
//void IDASolver<DAEParams, Scalar>::get_state_der(std::vector<realtype>& xp, std::vector<realtype>& zp) const
//{
//    realtype* yptr = N_VGetArrayPointer(yp_);
//
//    xp.assign(yptr, yptr + num_diffeq_);
//    zp.assign(yptr + num_diffeq_, yptr + N_);
//}

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

    for (size_t i = 0; i < udata->n; i++) {
        rval[i] = fval[i] - ypval[i];
    }

    for (size_t i = 0; i < udata->m; i++) {
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

    size_t N = u->n + u->p + u->q;
    size_t M = u->n + u->m;
    std::vector<realtype> X(yval, yval + (u->n + u->p));
    X.insert(X.end(), u->inputs->begin(), u->inputs->end()); // evaluate Jacobian using current inputs
    // augment
    auto jac = u->F.Jacobian(X);

    size_t JacN = N - u->q;
    for (size_t i = 0; i <= M; i++) {
        rowptrs[i] = i * JacN;
    }

    // jacobian of f
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < JacN; j++) {
            data[JacN * i + j] = jac[N * i + j];
            if (i < u->n and i == j) {
                data[JacN * i + j] += -cj;
            }
            colvals[JacN * i + j] = j;
        }
    }

    //for (size_t i = 0; i < jac.size(); i++) {
    //    if ((i % N) == 0 and i > 0) std::cout << std::endl;
    //    std::cout << jac[i] << "\t";
    //}
    //std::cout << std::endl;
    //SUNSparseMatrix_Print(JJ, stdout);

    return 0;
}

template <typename DAEParams, typename Scalar>
void IDASolver<DAEParams, Scalar>::print_jacobian() const
{
    realtype *yval;
    yval = N_VGetArrayPointer(yy_);
    std::vector<realtype> X(yval, yval + N_);
    X.insert(X.end(), u.begin(), u.end());
    auto jac = UserData_->F.Jacobian(X);
    for (size_t i = 0; i < jac.size(); i++) {
        if ((i % (N_ + num_inputs_)) == 0 and i > 0) std::cout << std::endl;
        std::cout << jac[i] << "\t";
    }
    std::cout << std::endl;
}

}
