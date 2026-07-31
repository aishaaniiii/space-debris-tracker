#pragma once 

#include <Eigen/Dense>

class KF //wrting a class so that it is the same as the python file 
{
    public:

    static const int iX = 0;
    static const int iV = 1;
    static const int n_var = iV + 1;

    using Vector = Eigen::Matrix<double, n_var, 1 >; //for the mean matrix
    using Matrix = Eigen::Matrix<double, n_var, n_var>; //for the covariance matrix
    
    KF(double x_init, double v_init, double a_var)
    : m_a_var(a_var)
    {
        m_mean(iX) = x_init;
        m_mean(iV) = v_init;

        m_cov.setIdentity();


    }

    void predict(double dt)
    {
        Matrix F;
        F.setIdentity();
        F(iX,iV) = dt;
        const Vector new_x = F * m_mean;

        Vector G;
        G(iX) = 0.5*dt*dt;
        G(iV) = dt;

        const Matrix new_P = F * m_cov * F + G * G.transpose()*m_a_var;
        m_cov = new_P;
        m_mean = new_x;
    }

    void update(double meas_val, double meas_var)
    {
        Eigen::Matrix<double, 1, n_var> H;
        H.setZero();
        H(0,iX) = 1;

        const double y = meas_val - H*m_mean;
        const double S = H * m_cov * H.transpose() + meas_var;

        const Vector K = m_cov * H.transpose() / S;

        Vector new_x = m_mean + K*y;
        Matrix new_P = (Matrix::Identity() - K*H)*m_cov;

        m_cov = new_P;
        m_mean = new_x;
    }

    Matrix cov() const
    {
        return m_cov;
    }

    Vector mean() const
    {
        return m_mean;
    }

    double pos() const
    {
        return m_mean(iX); //holder
    }

    double vel() const
    {
        return m_mean(iV); //holder
    }

    private:

    Vector m_mean;
    Matrix m_cov;
    const double m_a_var;

};