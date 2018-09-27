# Masters_Project

The Github Repository contains methods for solving the movement of a classical and quantum particle
in a 1-dimensional two state energy potential at finite temperatures. The mathematical techniques 
along with their numerical simulation were implemented in the classical and quantum functions libraries. 

The libraries contain functions to solve:

Classical:

- implementation of Verlet algorithm to solve Langevin equations of motion for individual trajectories.
- implementation of computing ``reactive flux" transmission and transition rates with averaging over
  individual trajectories.

Quantum:

- solving stochastic Schrodinger equation corresponding to the Caldeira-Leggett model in Lindblad form using
  four schemes:
	- Euler,
	- Heun,
	- Runge-Kutta,
	- Platen.
- solving Liouville equation for the density matrix resulting from the Caldeira-Leggett model in Lindblad
  form using superoperators.
- computing new proposed quantum transition rates.

The documentation of mathematical formulation can be found in the Mathematics folder in the pdf file. Please refer
to it for further explanation.