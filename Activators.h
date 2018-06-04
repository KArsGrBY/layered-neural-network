#ifndef NN1_ACTIVATORS_H
#define NN1_ACTIVATORS_H

#include "memory"
#include "functional"
#include "cmath"
#include <bits/stdc++.h>

struct Activator {
	std::function <double (double)> function;
	std::function <double (double)> derivative;

	Activator (std::function <double (double)> function, std::function <double (double)> derivative) {
		this->function = function;
		this->derivative = derivative;
	}

	Activator (std::function <double (double)> function) {
		this->function = function;
		this->derivative = function;
	}
};

namespace Activators {
	const std::shared_ptr <Activator> sigmoid = std::make_shared <Activator>([] (double x) {
																				 return 1 / (1 + exp(-x));
																			 },
																			 [] (double x) {
																				 return exp(-x) / pow(exp(-x) + 1, 2);
																			 });

	const std::shared_ptr <Activator> linear = std::make_shared <Activator>([] (double x) {
																				return x;
																			});

	const std::shared_ptr <Activator> unit = std::make_shared <Activator>([](double x) {
																				 return 1;
																			 });

	const std::shared_ptr <Activator> tanh = std::make_shared <Activator>([](double x) {
																				 return std::tanh(x);
																			 },
																			 [](double x) {
																				 return pow(cosh(x), -2);
																			 });
};

#endif //NN1_ACTIVATORS_H
