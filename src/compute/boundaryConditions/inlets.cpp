#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkos_types.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_CopyViews.hpp>

void constantVelocitySubsonicInlet(block_ b,
                                   face_ face,
                                   const std::function<void(block_, thtrdat_, int, std::string)> &eos,
                                   const thtrdat_ th,
                                   const std::string terms,
                                   const double /*tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int s0, s1, s2, plus;
  setHaloSlices(s0, s1, s2, plus, b.ni, b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getHaloSlice(b.q, face._nface, s1);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});

    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      s2 += plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);
      threeDsubview q2 = getHaloSlice(b.q, face._nface, s2);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // extrapolate pressure
            q0(i, j, 0) = 2.0 * q1(i, j, 0) - q2(i, j, 0);

            // apply velo in halo
            q0(i, j, 1) = face.qBcVals(i,j,1);
            q0(i, j, 2) = face.qBcVals(i,j,2);
            q0(i, j, 3) = face.qBcVals(i,j,3);

            // apply temperature in halo
            q0(i, j, 4) = face.qBcVals(i,j,4);

            // apply species in halo
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = face.qBcVals(i,j,n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("viscous") == 0) {

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(dqdx1.extent(0)),
                             static_cast<long>(dqdx1.extent(1)), b.ne});
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
      threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
      threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet viscous terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j, const int l) {
            // neumann all gradients
            dqdx0(i, j, l) = dqdx1(i, j, l);
            dqdy0(i, j, l) = dqdy1(i, j, l);
            dqdz0(i, j, l) = dqdz1(i, j, l);
          });
    }
  } else if (terms.compare("strict") == 0) {

  }
}

void cubicSplineSubsonicInlet(block_ b,
                              face_& face,
                              const std::function<void(block_, thtrdat_, int, std::string)> &eos,
                              const thtrdat_ th,
                              const std::string terms,
                              const double tme) {
  //-------------------------------------------------------------------------------------------|
  // Update target values with cubic spline, moving intervals if needed
  //-------------------------------------------------------------------------------------------|
  if (terms.compare("euler") == 0) {
    int totalIntervals = face.cubicSplineAlphas.extent(1);
    int currentInterval = static_cast<int>(std::floor(tme/face.intervalDt))%totalIntervals;

    // Check if the interval time puts us into the next interval
    if (currentInterval != face.currentInterval){
      face.currentInterval = currentInterval;
      // Now we update the interval alphas
      auto subview = Kokkos::subview(face.cubicSplineAlphas,
                                     Kokkos::ALL,
                                     face.currentInterval,
                                     Kokkos::ALL,
                                     Kokkos::ALL,
                                     Kokkos::ALL);

      auto intervalAlphasMirror = Kokkos::create_mirror_view(face.intervalAlphas);
      Kokkos::deep_copy(intervalAlphasMirror,subview);
      Kokkos::deep_copy(face.intervalAlphas, intervalAlphasMirror);

    }
    // Now we comute the target values

    MDRange2 range_face = MDRange2({0, 0}, {face.intervalAlphas.extent(1)-1,
                                            face.intervalAlphas.extent(2)-1});
    double interpTime = tme - static_cast<double>(
                              static_cast<int>(tme/face.intervalDt/totalIntervals) *
                                                                   totalIntervals) * face.intervalDt;
    double intervalTime = interpTime - face.intervalDt*static_cast<double>(face.currentInterval);
    // Form of the cubic spline for the interval "i" is

    // u(t) = alpha[0]*(t-t[i-1])**3 + alpha[1]*(t-t[i-1])**2 + alpha[2]*(t-t[i-1]) + alpha[3]

    // where t[i-1] is the value of time at the beginning of the current interval,
    // i.e. if we are in interval [3] then t[3-1] is the value of time for frame 3.
    //
    //| frame 0 |              | frame 1 |              | frame 2 |              | frame 3 |
    //|   t[0]  | -----------> |   t[1]  | -----------> |   t[2]  | -----------> |   t[3]  |
    //|         | <interval 0> |         | <interval 1> |         | <interval 2> |         |
    Kokkos::parallel_for(
        "Cubic spline subsonic", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          face.qBcVals(i,j,1) = 0.0;
          face.qBcVals(i,j,2) = 0.0;
          face.qBcVals(i,j,3) = 0.0;
          for (int k=0; k < 4; k++) {
            face.qBcVals(i,j,1) += face.intervalAlphas(k,i,j,0)*pow(intervalTime,static_cast<double>(3-k));
            face.qBcVals(i,j,2) += face.intervalAlphas(k,i,j,1)*pow(intervalTime,static_cast<double>(3-k));
            face.qBcVals(i,j,3) += face.intervalAlphas(k,i,j,2)*pow(intervalTime,static_cast<double>(3-k));
          }
        });
      // Now we call constant velo subsonic bc as usual
      constantVelocitySubsonicInlet(b, face, eos, th, terms, tme);
      }else{
      constantVelocitySubsonicInlet(b, face, eos, th, terms, tme);
  }
}

void supersonicInlet(block_ b,
                     face_ face,
                     const std::function<void(block_, thtrdat_, int, std::string)> &eos,
                     const thtrdat_ th,
                     const std::string terms,
                     const double /*tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int s0, s1, s2, plus;
  setHaloSlices(s0, s1, s2, plus, b.ni, b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getHaloSlice(b.q, face._nface, s1);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});

    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      s2 += plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);
      threeDsubview q2 = getHaloSlice(b.q, face._nface, s2);

      Kokkos::parallel_for(
          "Supersonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // apply pressure on face
            q0(i, j, 0) = face.qBcVals(i,j,0);

            // apply velo on face
            q0(i, j, 1) = face.qBcVals(i,j,1);
            q0(i, j, 2) = face.qBcVals(i,j,2);
            q0(i, j, 3) = face.qBcVals(i,j,3);

            // apply temperature on face
            q0(i, j, 4) = face.qBcVals(i,j,4);

            // apply species on face
            // TODO: This is an unprotected extrapolation.
            // Is this a good thing to be doing?
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = face.qBcVals(i,j,n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("viscous") == 0) {

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(dqdx1.extent(0)),
                             static_cast<long>(dqdx1.extent(1)), b.ne});
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
      threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
      threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

      Kokkos::parallel_for(
          "Supersonic inlet viscous terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j, const int l) {
            // neumann all gradients
            dqdx0(i, j, l) = dqdx1(i, j, l);
            dqdy0(i, j, l) = dqdy1(i, j, l);
            dqdz0(i, j, l) = dqdz1(i, j, l);
          });
    }
  } else if (terms.compare("strict") == 0) {

  }
}

void constantMassFluxSubsonicInlet(block_ b,
                                   face_ face,
                                   const std::function<void(block_, thtrdat_, int, std::string)> &eos,
                                   const thtrdat_ th,
                                   const std::string terms,
                                   const double /*tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int s0, s1, s2, plus;
  setHaloSlices(s0, s1, s2, plus, b.ni, b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getHaloSlice(b.q, face._nface, s1);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});

    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      s2 += plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);
      threeDsubview q2 = getHaloSlice(b.q, face._nface, s2);

      Kokkos::parallel_for(
          "Constant mass flux subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // extrapolate pressure
            q0(i, j, 0) = 2.0 * q1(i, j, 0) - q2(i, j, 0);

            // apply zero velo on face to make subsequent updates easier
            q0(i, j, 1) = 0.0;
            q0(i, j, 2) = 0.0;
            q0(i, j, 3) = 0.0;

            // apply temperature on face
            q0(i, j, 4) = face.qBcVals(i,j,4);

            // apply species on face
            // TODO: This is an unprotected extrapolation.
            // Is this a good thing to be doing?
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = face.qBcVals(i,j,n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

    // We now have a valid density value
    // set momentums, and velocities to match the desired mass flux
    // NOTE: We have to be careful with the indexing to accomodate fourth
    // order. In particular, we cannot just use s1 for all the extrapolations
    // so we have to make s2 start with s1 then increment

    // Reset first slice indicies, and make s2 start at s1
    s0 += plus * (ng-1);
    s2 -= plus * (ng-1);
    s2 -= plus ;

    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      s2 += plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);
      threeDsubview q2 = getHaloSlice(b.q, face._nface, s2);
      threeDsubview Q0 = getHaloSlice(b.Q, face._nface, s0);
      threeDsubview Q2 = getHaloSlice(b.Q, face._nface, s2);

      Kokkos::parallel_for(
          "Constant mass flux subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // Target rhoU
            double &rhou = face.QBcVals(i,j,1);
            double &rhov = face.QBcVals(i,j,2);
            double &rhow = face.QBcVals(i,j,3);

            // Set the velocities in the halo such that
            // 1/2(rho1+rho2)*1/2(u1+u2) evaluates to our desired rhou
            q0(i, j, 1) =
                4.0 * rhou / (Q0(i, j, 0) + Q2(i, j, 0)) - q2(i, j, 1);
            q0(i, j, 2) =
                4.0 * rhov / (Q0(i, j, 0) + Q2(i, j, 0)) - q2(i, j, 2);
            q0(i, j, 3) =
                4.0 * rhow / (Q0(i, j, 0) + Q2(i, j, 0)) - q2(i, j, 3);

            // update momentum
            double &rho = Q0(i, j, 0);
            Q0(i, j, 1) = q0(i, j, 1) * rho;
            Q0(i, j, 2) = q0(i, j, 2) * rho;
            Q0(i, j, 3) = q0(i, j, 3) * rho;

            // we have created tke in halo, compute that and add it to
            // the existing rhoE, which is just internal energy at this point
            double tke = 0.5 *
                         (pow(q0(i, j, 1), 2.0) + pow(q0(i, j, 2), 2.0) +
                          pow(q0(i, j, 3), 2.0)) *
                         rho;
            Q0(i, j, 4) += tke;
          });
    }

  } else if (terms.compare("viscous") == 0) {

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(dqdx1.extent(0)),
                             static_cast<long>(dqdx1.extent(1)), b.ne});
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
      threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
      threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

      Kokkos::parallel_for(
          "Constant mass flux subsonic inlet viscous terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j, const int l) {
            // neumann all gradients
            dqdx0(i, j, l) = dqdx1(i, j, l);
            dqdy0(i, j, l) = dqdy1(i, j, l);
            dqdz0(i, j, l) = dqdz1(i, j, l);
          });
    }
  } else if (terms.compare("strict") == 0) {

    int slc;
    threeDsubview iF;
    twoDsubview iS;
    switch(face._nface) {
      case 1:
        slc = s1;
        iF = getHaloSlice(b.iF, face._nface, slc);
        iS = getHaloSlice(b.iS, face._nface, slc);
        break;
      case 3:
        slc = s1;
        iF = getHaloSlice(b.jF, face._nface, slc);
        iS = getHaloSlice(b.jS, face._nface, slc);
        break;
      case 5:
        slc = s1;
        iF = getHaloSlice(b.kF, face._nface, slc);
        iS = getHaloSlice(b.kS, face._nface, slc);
        break;
      case 2:
        slc = s0;
        iF = getHaloSlice(b.iF, face._nface, slc);
        iS = getHaloSlice(b.iS, face._nface, slc);
        break;
      case 4:
        slc = s0;
        iF = getHaloSlice(b.jF, face._nface, slc);
        iS = getHaloSlice(b.jS, face._nface, slc);
        break;
      case 6:
        slc = s0;
        iF = getHaloSlice(b.kF, face._nface, slc);
        iS = getHaloSlice(b.kS, face._nface, slc);
        break;
    }
    MDRange2 range_face = MDRange2({0, 0}, {iF.extent(0), iF.extent(1)});
      Kokkos::parallel_for(
          "Strict convective flux specification", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // explicitely set the mass flux
            double &rhou = face.QBcVals(i,j,1);
            double &rhov = face.QBcVals(i,j,2);
            double &rhow = face.QBcVals(i,j,3);

            double mDotPerUnitArea = sqrt(pow(rhou,2.0) + pow(rhov,2.0) + pow(rhow,2.0));

            iF(i, j, 0) = mDotPerUnitArea * iS(i,j);
            for (int n = 5; n < b.ne; n++) {
              double &Y = face.qBcVals(i,j,n);
              iF(i,j,n) = mDotPerUnitArea * Y * iS(i,j);
            }
          });
  }
}
