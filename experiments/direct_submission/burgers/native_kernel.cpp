// Computational acceleration of the already executed five-entry row update.
// No model, truth, controller, or reference information enters this kernel.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

extern "C" int direct_rows_step(double* q_io, double* R_io, const double* v,
                               int count, double dt, double tau, double gain,
                               int adapt, int* steps_out,
                               char* error, int error_capacity) {
    auto fail = [&](const char* message) {
        if (error && error_capacity > 0) std::snprintf(error, error_capacity, "%s", message);
        return 1;
    };
    try {
        if (count <= 0 || !q_io || !R_io || !v || !steps_out ||
            !std::isfinite(dt) || dt <= 0. || !std::isfinite(tau) || tau <= 0. ||
            !std::isfinite(gain) || gain < 0.) return fail("invalid kernel arguments");
        constexpr int D = 5;
        const int nq = count * D, nr = count * D * D;
        std::vector<double> q(q_io, q_io+nq), old(R_io, R_io+nr), outer(nr), end(nr);
        for (int n=0; n<count; ++n) {
            for (int i=0; i<D; ++i) {
                if (!std::isfinite(v[n*D+i])) return fail("finite raw-context secants required");
                for (int j=0; j<D; ++j) outer[n*D*D+i*D+j] = v[n*D+i]*v[n*D+j];
            }
        }
        const double endpoint_decay = std::exp(-dt/tau);
        const double endpoint_weight = -tau*std::expm1(-dt/tau);
        for (int i=0; i<nr; ++i) end[i] = endpoint_decay*old[i]+endpoint_weight*outer[i];
        if (!adapt) {
            std::copy(end.begin(), end.end(), R_io);
            *steps_out = 0;
            return 0;
        }
        double bound = 0.;
        for (int n=0; n<count; ++n) {
            double trace_old = 0., trace_end = 0.;
            for (int i=0; i<D; ++i) {
                trace_old += old[n*D*D+i*D+i];
                trace_end += end[n*D*D+i*D+i];
            }
            bound = std::max(bound, std::max(trace_old, trace_end));
        }
        const double needed = std::ceil(gain*dt*bound/.25);
        if (!std::isfinite(needed) || needed > 10000.)
            return fail("10000-subdivision per-sample numerical work cap exceeded");
        const int steps = std::max(4, static_cast<int>(needed));
        const double ds = dt/steps;
        std::vector<double> k1(nq), k2(nq), k3(nq), k4(nq), temp(nq);
        auto tangent = [&](const std::vector<double>& x, double time, std::vector<double>& out) {
            const double decay = std::exp(-time/tau);
            const double weight = -tau*std::expm1(-time/tau);
            for (int n=0; n<count; ++n) {
                double qr[D] = {}, scalar = 0.;
                for (int j=0; j<D; ++j) {
                    for (int i=0; i<D; ++i) {
                        const int index = n*D*D+i*D+j;
                        qr[j] += x[n*D+i]*(decay*old[index]+weight*outer[index]);
                    }
                    scalar += qr[j]*x[n*D+j];
                }
                for (int j=0; j<D; ++j) out[n*D+j] = -gain*(qr[j]-scalar*x[n*D+j]);
            }
        };
        for (int k=0; k<steps; ++k) {
            const double time = k*ds;
            tangent(q, time, k1);
            for (int i=0; i<nq; ++i) temp[i] = q[i]+.5*ds*k1[i];
            tangent(temp, time+.5*ds, k2);
            for (int i=0; i<nq; ++i) temp[i] = q[i]+.5*ds*k2[i];
            tangent(temp, time+.5*ds, k3);
            for (int i=0; i<nq; ++i) temp[i] = q[i]+ds*k3[i];
            tangent(temp, time+ds, k4);
            for (int n=0; n<count; ++n) {
                double norm_squared = 0.;
                for (int j=0; j<D; ++j) {
                    const int i = n*D+j;
                    q[i] += ds*(k1[i]+2.*k2[i]+2.*k3[i]+k4[i])/6.;
                    norm_squared += q[i]*q[i];
                }
                const double norm = std::sqrt(norm_squared);
                if (!std::isfinite(norm) || norm < 1e-15)
                    return fail("nonfinite or vanishing generic row");
                for (int j=0; j<D; ++j) q[n*D+j] /= norm;
            }
        }
        std::copy(q.begin(), q.end(), q_io);
        std::copy(end.begin(), end.end(), R_io);
        *steps_out = steps;
        return 0;
    } catch (const std::exception& ex) {
        return fail(ex.what());
    } catch (...) {
        return fail("unknown native kernel exception");
    }
}
