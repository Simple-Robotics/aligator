# Formulation

The objective of this library is to model and solve optimal control problems (OCPs) of the form

$$
\begin{aligned}
    \min_{x,u}~& \int_0^T \ell(x, u)\, dt + \ell_\mathrm{f}(x(T)) \\\\
    \subjectto  & \\dot{x}(t) = f(x(t), u(t)) \\\\
                & g(x(t), u(t)) = 0 \\\\
                & h(x(t), u(t)) \leq 0
\end{aligned}
$$

## Transcription

Consider the transcription with implicit-form discrete dynamics:
$$
\begin{aligned}
    \min_{\bfx,\bfu}~& J(\bfx, \bfu) = \sum_{i=0}^{N-1} \ell_i(x_i, u_i) + \ell_N(x_N) \\\\
    \subjectto  & f(x_i, u_i, x_{i+1}) = 0 \\\\
                & g(x_i, u_i) = 0 \\\\
                & h(x_i, u_i) \leq 0
\end{aligned}
$$

## Docs: Known issues

Under `doxygen==1.9.2`, class collaboration diagrams will *not* pick up relationships stemming from `shared_ptr<T>` members.
