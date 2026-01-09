/// @file
/// @copyright Copyright (C) 2026 INRIA
#pragma once

#include <boost/version.hpp>
#if defined(BOOST_VERSION) && BOOST_VERSION >= 107600
#include <boost/core/make_span.hpp>
#else
#include "aligator/compat/boost/core/make_span.hpp"
#endif
