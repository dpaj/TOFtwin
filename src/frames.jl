using StaticArrays
using LinearAlgebra
import LinearAlgebra: inv   # <-- add this line

const Vec3 = SVector{3,Float64}
const Mat3 = SMatrix{3,3,Float64,9}

"Rigid transform: p_to = R*p_from + t"
struct Rigid
    R::Mat3
    t::Vec3
end

Rigid() = Rigid(Mat3(I), Vec3(0.0, 0.0, 0.0))

# composition: A∘B means apply B then A
function (A::Rigid)(B::Rigid)
    R = A.R * B.R
    t = A.R * B.t + A.t
    return Rigid(R, t)
end

apply_point(T::Rigid, p::Vec3) = T.R*p + T.t
apply_vec(T::Rigid, v::Vec3)   = T.R*v
inv(T::Rigid) = Rigid(transpose(T.R), -(transpose(T.R)*T.t))

# basic rotations (right-handed)
Rx(a) = Mat3([1.0 0.0 0.0;
              0.0 cos(a) -sin(a);
              0.0 sin(a)  cos(a)])

Ry(a) = Mat3([ cos(a) 0.0 sin(a);
               0.0    1.0 0.0;
              -sin(a) 0.0 cos(a)])

Rz(a) = Mat3([cos(a) -sin(a) 0.0;
              sin(a)  cos(a) 0.0;
              0.0     0.0    1.0])

rigid(R::Mat3; t::Vec3=Vec3(0.0,0.0,0.0)) = Rigid(R, t)
