#ifndef __boundaryConditions_H__
#define __boundaryConditions_H__

#include "block_.hpp"
#include "face_.hpp"
#include "thtrdat_.hpp"

// ./boundaryConditions
//    |------> inlets
void constantVelocitySubsonicInlet(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/);
void cubicSplineSubsonicInlet(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double &tme);
void supersonicInlet(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/);
void constantMassFluxSubsonicInlet(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/);
//    |------> walls
void adiabaticNoSlipWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/);
void adiabaticSlipWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/);
void adiabaticMovingWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/);
void isoTNoSlipWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/);
void isoTSlipWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/);
void isoTMovingWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/);
//    |------> exits
void constantPressureSubsonicExit(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/);
void supersonicExit(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/);
//    |------> periodics
void periodicRotHigh(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> /*&eos*/,
    const thtrdat_ /*&th*/, const std::string &terms, const double /*&tme*/);
void periodicRotLow(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> /*&eos*/,
    const thtrdat_ /*&th*/, const std::string &terms, const double /*&tme*/);

#endif
