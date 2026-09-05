"""Corrected-K2 plus six-layer structural 32k antithetic Sobol sampling.

A fixed Gaussian point set is generated once in setup(). The final six layers
use conservative tapered dead/on/kink thresholds; the final two also collapse
always-on paths analytically.
"""
from __future__ import annotations

import random
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

K2_COEFFICIENT = 0.9918858550287327
LAMBDA = 0.9768709009656374
N_SAMPLES = 32_768
HALF_SAMPLES = N_SAMPLES // 2
STRUCTURAL_THRESHOLDS = (3.5, 3.4, 3.3, 3.2, 3.0, 3.0)
SOBOL_BITS = 32
SOBOL_SCALE = float(1 << SOBOL_BITS)
SOBOL_SEED_OFFSET = 8_675_309
FIXED_POINTSET_SEED = 0
SAMPLES_FILE = "samples.npz"
_COV_RESCALE_THRESHOLD = 1e100

# Joe-Kuo new-joe-kuo-6.21201 parameters for dimensions 2 through 256.
# Each row is (primitive-polynomial degree, coefficient, initial values).
_DIRECTION_ROWS = (
    (1, 0, (1,)),
    (2, 1, (1, 3,)),
    (3, 1, (1, 3, 1,)),
    (3, 2, (1, 1, 1,)),
    (4, 1, (1, 1, 3, 3,)),
    (4, 4, (1, 3, 5, 13,)),
    (5, 2, (1, 1, 5, 5, 17,)),
    (5, 4, (1, 1, 5, 5, 5,)),
    (5, 7, (1, 1, 7, 11, 19,)),
    (5, 11, (1, 1, 5, 1, 1,)),
    (5, 13, (1, 1, 1, 3, 11,)),
    (5, 14, (1, 3, 5, 5, 31,)),
    (6, 1, (1, 3, 3, 9, 7, 49,)),
    (6, 13, (1, 1, 1, 15, 21, 21,)),
    (6, 16, (1, 3, 1, 13, 27, 49,)),
    (6, 19, (1, 1, 1, 15, 7, 5,)),
    (6, 22, (1, 3, 1, 15, 13, 25,)),
    (6, 25, (1, 1, 5, 5, 19, 61,)),
    (7, 1, (1, 3, 7, 11, 23, 15, 103,)),
    (7, 4, (1, 3, 7, 13, 13, 15, 69,)),
    (7, 7, (1, 1, 3, 13, 7, 35, 63,)),
    (7, 8, (1, 3, 5, 9, 1, 25, 53,)),
    (7, 14, (1, 3, 1, 13, 9, 35, 107,)),
    (7, 19, (1, 3, 1, 5, 27, 61, 31,)),
    (7, 21, (1, 1, 5, 11, 19, 41, 61,)),
    (7, 28, (1, 3, 5, 3, 3, 13, 69,)),
    (7, 31, (1, 1, 7, 13, 1, 19, 1,)),
    (7, 32, (1, 3, 7, 5, 13, 19, 59,)),
    (7, 37, (1, 1, 3, 9, 25, 29, 41,)),
    (7, 41, (1, 3, 5, 13, 23, 1, 55,)),
    (7, 42, (1, 3, 7, 3, 13, 59, 17,)),
    (7, 50, (1, 3, 1, 3, 5, 53, 69,)),
    (7, 55, (1, 1, 5, 5, 23, 33, 13,)),
    (7, 56, (1, 1, 7, 7, 1, 61, 123,)),
    (7, 59, (1, 1, 7, 9, 13, 61, 49,)),
    (7, 62, (1, 3, 3, 5, 3, 55, 33,)),
    (8, 14, (1, 3, 1, 15, 31, 13, 49, 245,)),
    (8, 21, (1, 3, 5, 15, 31, 59, 63, 97,)),
    (8, 22, (1, 3, 1, 11, 11, 11, 77, 249,)),
    (8, 38, (1, 3, 1, 11, 27, 43, 71, 9,)),
    (8, 47, (1, 1, 7, 15, 21, 11, 81, 45,)),
    (8, 49, (1, 3, 7, 3, 25, 31, 65, 79,)),
    (8, 50, (1, 3, 1, 1, 19, 11, 3, 205,)),
    (8, 52, (1, 1, 5, 9, 19, 21, 29, 157,)),
    (8, 56, (1, 3, 7, 11, 1, 33, 89, 185,)),
    (8, 67, (1, 3, 3, 3, 15, 9, 79, 71,)),
    (8, 70, (1, 3, 7, 11, 15, 39, 119, 27,)),
    (8, 84, (1, 1, 3, 1, 11, 31, 97, 225,)),
    (8, 97, (1, 1, 1, 3, 23, 43, 57, 177,)),
    (8, 103, (1, 3, 7, 7, 17, 17, 37, 71,)),
    (8, 115, (1, 3, 1, 5, 27, 63, 123, 213,)),
    (8, 122, (1, 1, 3, 5, 11, 43, 53, 133,)),
    (9, 8, (1, 3, 5, 5, 29, 17, 47, 173, 479,)),
    (9, 13, (1, 3, 3, 11, 3, 1, 109, 9, 69,)),
    (9, 16, (1, 1, 1, 5, 17, 39, 23, 5, 343,)),
    (9, 22, (1, 3, 1, 5, 25, 15, 31, 103, 499,)),
    (9, 25, (1, 1, 1, 11, 11, 17, 63, 105, 183,)),
    (9, 44, (1, 1, 5, 11, 9, 29, 97, 231, 363,)),
    (9, 47, (1, 1, 5, 15, 19, 45, 41, 7, 383,)),
    (9, 52, (1, 3, 7, 7, 31, 19, 83, 137, 221,)),
    (9, 55, (1, 1, 1, 3, 23, 15, 111, 223, 83,)),
    (9, 59, (1, 1, 5, 13, 31, 15, 55, 25, 161,)),
    (9, 62, (1, 1, 3, 13, 25, 47, 39, 87, 257,)),
    (9, 67, (1, 1, 1, 11, 21, 53, 125, 249, 293,)),
    (9, 74, (1, 1, 7, 11, 11, 7, 57, 79, 323,)),
    (9, 81, (1, 1, 5, 5, 17, 13, 81, 3, 131,)),
    (9, 82, (1, 1, 7, 13, 23, 7, 65, 251, 475,)),
    (9, 87, (1, 3, 5, 1, 9, 43, 3, 149, 11,)),
    (9, 91, (1, 1, 3, 13, 31, 13, 13, 255, 487,)),
    (9, 94, (1, 3, 3, 1, 5, 63, 89, 91, 127,)),
    (9, 103, (1, 1, 3, 3, 1, 19, 123, 127, 237,)),
    (9, 104, (1, 1, 5, 7, 23, 31, 37, 243, 289,)),
    (9, 109, (1, 1, 5, 11, 17, 53, 117, 183, 491,)),
    (9, 122, (1, 1, 1, 5, 1, 13, 13, 209, 345,)),
    (9, 124, (1, 1, 3, 15, 1, 57, 115, 7, 33,)),
    (9, 137, (1, 3, 1, 11, 7, 43, 81, 207, 175,)),
    (9, 138, (1, 3, 1, 1, 15, 27, 63, 255, 49,)),
    (9, 143, (1, 3, 5, 3, 27, 61, 105, 171, 305,)),
    (9, 145, (1, 1, 5, 3, 1, 3, 57, 249, 149,)),
    (9, 152, (1, 1, 3, 5, 5, 57, 15, 13, 159,)),
    (9, 157, (1, 1, 1, 11, 7, 11, 105, 141, 225,)),
    (9, 167, (1, 3, 3, 5, 27, 59, 121, 101, 271,)),
    (9, 173, (1, 3, 5, 9, 11, 49, 51, 59, 115,)),
    (9, 176, (1, 1, 7, 1, 23, 45, 125, 71, 419,)),
    (9, 181, (1, 1, 3, 5, 23, 5, 105, 109, 75,)),
    (9, 182, (1, 1, 7, 15, 7, 11, 67, 121, 453,)),
    (9, 185, (1, 3, 7, 3, 9, 13, 31, 27, 449,)),
    (9, 191, (1, 3, 1, 15, 19, 39, 39, 89, 15,)),
    (9, 194, (1, 1, 1, 1, 1, 33, 73, 145, 379,)),
    (9, 199, (1, 3, 1, 15, 15, 43, 29, 13, 483,)),
    (9, 218, (1, 1, 7, 3, 19, 27, 85, 131, 431,)),
    (9, 220, (1, 3, 3, 3, 5, 35, 23, 195, 349,)),
    (9, 227, (1, 3, 3, 7, 9, 27, 39, 59, 297,)),
    (9, 229, (1, 1, 3, 9, 11, 17, 13, 241, 157,)),
    (9, 230, (1, 3, 7, 15, 25, 57, 33, 189, 213,)),
    (9, 234, (1, 1, 7, 1, 9, 55, 73, 83, 217,)),
    (9, 236, (1, 3, 3, 13, 19, 27, 23, 113, 249,)),
    (9, 241, (1, 3, 5, 3, 23, 43, 3, 253, 479,)),
    (9, 244, (1, 1, 5, 5, 11, 5, 45, 117, 217,)),
    (9, 253, (1, 3, 3, 7, 29, 37, 33, 123, 147,)),
    (10, 4, (1, 3, 1, 15, 5, 5, 37, 227, 223, 459,)),
    (10, 13, (1, 1, 7, 5, 5, 39, 63, 255, 135, 487,)),
    (10, 19, (1, 3, 1, 7, 9, 7, 87, 249, 217, 599,)),
    (10, 22, (1, 1, 3, 13, 9, 47, 7, 225, 363, 247,)),
    (10, 50, (1, 3, 7, 13, 19, 13, 9, 67, 9, 737,)),
    (10, 55, (1, 3, 5, 5, 19, 59, 7, 41, 319, 677,)),
    (10, 64, (1, 1, 5, 3, 31, 63, 15, 43, 207, 789,)),
    (10, 69, (1, 1, 7, 9, 13, 39, 3, 47, 497, 169,)),
    (10, 98, (1, 3, 1, 7, 21, 17, 97, 19, 415, 905,)),
    (10, 107, (1, 3, 7, 1, 3, 31, 71, 111, 165, 127,)),
    (10, 115, (1, 1, 5, 11, 1, 61, 83, 119, 203, 847,)),
    (10, 121, (1, 3, 3, 13, 9, 61, 19, 97, 47, 35,)),
    (10, 127, (1, 1, 7, 7, 15, 29, 63, 95, 417, 469,)),
    (10, 134, (1, 3, 1, 9, 25, 9, 71, 57, 213, 385,)),
    (10, 140, (1, 3, 5, 13, 31, 47, 101, 57, 39, 341,)),
    (10, 145, (1, 1, 3, 3, 31, 57, 125, 173, 365, 551,)),
    (10, 152, (1, 3, 7, 1, 13, 57, 67, 157, 451, 707,)),
    (10, 158, (1, 1, 1, 7, 21, 13, 105, 89, 429, 965,)),
    (10, 161, (1, 1, 5, 9, 17, 51, 45, 119, 157, 141,)),
    (10, 171, (1, 3, 7, 7, 13, 45, 91, 9, 129, 741,)),
    (10, 181, (1, 3, 7, 1, 23, 57, 67, 141, 151, 571,)),
    (10, 194, (1, 1, 3, 11, 17, 47, 93, 107, 375, 157,)),
    (10, 199, (1, 3, 3, 5, 11, 21, 43, 51, 169, 915,)),
    (10, 203, (1, 1, 5, 3, 15, 55, 101, 67, 455, 625,)),
    (10, 208, (1, 3, 5, 9, 1, 23, 29, 47, 345, 595,)),
    (10, 227, (1, 3, 7, 7, 5, 49, 29, 155, 323, 589,)),
    (10, 242, (1, 3, 3, 7, 5, 41, 127, 61, 261, 717,)),
    (10, 251, (1, 3, 7, 7, 17, 23, 117, 67, 129, 1009,)),
    (10, 253, (1, 1, 3, 13, 11, 39, 21, 207, 123, 305,)),
    (10, 265, (1, 1, 3, 9, 29, 3, 95, 47, 231, 73,)),
    (10, 266, (1, 3, 1, 9, 1, 29, 117, 21, 441, 259,)),
    (10, 274, (1, 3, 1, 13, 21, 39, 125, 211, 439, 723,)),
    (10, 283, (1, 1, 7, 3, 17, 63, 115, 89, 49, 773,)),
    (10, 289, (1, 3, 7, 13, 11, 33, 101, 107, 63, 73,)),
    (10, 295, (1, 1, 5, 5, 13, 57, 63, 135, 437, 177,)),
    (10, 301, (1, 1, 3, 7, 27, 63, 93, 47, 417, 483,)),
    (10, 316, (1, 1, 3, 1, 23, 29, 1, 191, 49, 23,)),
    (10, 319, (1, 1, 3, 15, 25, 55, 9, 101, 219, 607,)),
    (10, 324, (1, 3, 1, 7, 7, 19, 51, 251, 393, 307,)),
    (10, 346, (1, 3, 3, 3, 25, 55, 17, 75, 337, 3,)),
    (10, 352, (1, 1, 1, 13, 25, 17, 65, 45, 479, 413,)),
    (10, 361, (1, 1, 7, 7, 27, 49, 99, 161, 213, 727,)),
    (10, 367, (1, 3, 5, 1, 23, 5, 43, 41, 251, 857,)),
    (10, 382, (1, 3, 3, 7, 11, 61, 39, 87, 383, 835,)),
    (10, 395, (1, 1, 3, 15, 13, 7, 29, 7, 505, 923,)),
    (10, 398, (1, 3, 7, 1, 5, 31, 47, 157, 445, 501,)),
    (10, 400, (1, 1, 3, 7, 1, 43, 9, 147, 115, 605,)),
    (10, 412, (1, 3, 3, 13, 5, 1, 119, 211, 455, 1001,)),
    (10, 419, (1, 1, 3, 5, 13, 19, 3, 243, 75, 843,)),
    (10, 422, (1, 3, 7, 7, 1, 19, 91, 249, 357, 589,)),
    (10, 426, (1, 1, 1, 9, 1, 25, 109, 197, 279, 411,)),
    (10, 428, (1, 3, 1, 15, 23, 57, 59, 135, 191, 75,)),
    (10, 433, (1, 1, 5, 15, 29, 21, 39, 253, 383, 349,)),
    (10, 446, (1, 3, 3, 5, 19, 45, 61, 151, 199, 981,)),
    (10, 454, (1, 3, 5, 13, 9, 61, 107, 141, 141, 1,)),
    (10, 457, (1, 3, 1, 11, 27, 25, 85, 105, 309, 979,)),
    (10, 472, (1, 3, 3, 11, 19, 7, 115, 223, 349, 43,)),
    (10, 493, (1, 1, 7, 9, 21, 39, 123, 21, 275, 927,)),
    (10, 505, (1, 1, 7, 13, 15, 41, 47, 243, 303, 437,)),
    (10, 508, (1, 1, 1, 7, 7, 3, 15, 99, 409, 719,)),
    (11, 2, (1, 3, 3, 15, 27, 49, 113, 123, 113, 67, 469,)),
    (11, 11, (1, 3, 7, 11, 3, 23, 87, 169, 119, 483, 199,)),
    (11, 21, (1, 1, 5, 15, 7, 17, 109, 229, 179, 213, 741,)),
    (11, 22, (1, 1, 5, 13, 11, 17, 25, 135, 403, 557, 1433,)),
    (11, 35, (1, 3, 1, 1, 1, 61, 67, 215, 189, 945, 1243,)),
    (11, 49, (1, 1, 7, 13, 17, 33, 9, 221, 429, 217, 1679,)),
    (11, 50, (1, 1, 3, 11, 27, 3, 15, 93, 93, 865, 1049,)),
    (11, 56, (1, 3, 7, 7, 25, 41, 121, 35, 373, 379, 1547,)),
    (11, 61, (1, 3, 3, 9, 11, 35, 45, 205, 241, 9, 59,)),
    (11, 70, (1, 3, 1, 7, 3, 51, 7, 177, 53, 975, 89,)),
    (11, 74, (1, 1, 3, 5, 27, 1, 113, 231, 299, 759, 861,)),
    (11, 79, (1, 3, 3, 15, 25, 29, 5, 255, 139, 891, 2031,)),
    (11, 84, (1, 3, 1, 1, 13, 9, 109, 193, 419, 95, 17,)),
    (11, 88, (1, 1, 7, 9, 3, 7, 29, 41, 135, 839, 867,)),
    (11, 103, (1, 1, 7, 9, 25, 49, 123, 217, 113, 909, 215,)),
    (11, 104, (1, 1, 7, 3, 23, 15, 43, 133, 217, 327, 901,)),
    (11, 112, (1, 1, 3, 3, 13, 53, 63, 123, 477, 711, 1387,)),
    (11, 115, (1, 1, 3, 15, 7, 29, 75, 119, 181, 957, 247,)),
    (11, 117, (1, 1, 1, 11, 27, 25, 109, 151, 267, 99, 1461,)),
    (11, 122, (1, 3, 7, 15, 5, 5, 53, 145, 11, 725, 1501,)),
    (11, 134, (1, 3, 7, 1, 9, 43, 71, 229, 157, 607, 1835,)),
    (11, 137, (1, 3, 3, 13, 25, 1, 5, 27, 471, 349, 127,)),
    (11, 146, (1, 1, 1, 1, 23, 37, 9, 221, 269, 897, 1685,)),
    (11, 148, (1, 1, 3, 3, 31, 29, 51, 19, 311, 553, 1969,)),
    (11, 157, (1, 3, 7, 5, 5, 55, 17, 39, 475, 671, 1529,)),
    (11, 158, (1, 1, 7, 1, 1, 35, 47, 27, 437, 395, 1635,)),
    (11, 162, (1, 1, 7, 3, 13, 23, 43, 135, 327, 139, 389,)),
    (11, 164, (1, 3, 7, 3, 9, 25, 91, 25, 429, 219, 513,)),
    (11, 168, (1, 1, 3, 5, 13, 29, 119, 201, 277, 157, 2043,)),
    (11, 173, (1, 3, 5, 3, 29, 57, 13, 17, 167, 739, 1031,)),
    (11, 185, (1, 3, 3, 5, 29, 21, 95, 27, 255, 679, 1531,)),
    (11, 186, (1, 3, 7, 15, 9, 5, 21, 71, 61, 961, 1201,)),
    (11, 191, (1, 3, 5, 13, 15, 57, 33, 93, 459, 867, 223,)),
    (11, 193, (1, 1, 1, 15, 17, 43, 127, 191, 67, 177, 1073,)),
    (11, 199, (1, 1, 1, 15, 23, 7, 21, 199, 75, 293, 1611,)),
    (11, 213, (1, 3, 7, 13, 15, 39, 21, 149, 65, 741, 319,)),
    (11, 214, (1, 3, 7, 11, 23, 13, 101, 89, 277, 519, 711,)),
    (11, 220, (1, 3, 7, 15, 19, 27, 85, 203, 441, 97, 1895,)),
    (11, 227, (1, 3, 1, 3, 29, 25, 21, 155, 11, 191, 197,)),
    (11, 236, (1, 1, 7, 5, 27, 11, 81, 101, 457, 675, 1687,)),
    (11, 242, (1, 3, 1, 5, 25, 5, 65, 193, 41, 567, 781,)),
    (11, 251, (1, 3, 1, 5, 11, 15, 113, 77, 411, 695, 1111,)),
    (11, 256, (1, 1, 3, 9, 11, 53, 119, 171, 55, 297, 509,)),
    (11, 259, (1, 1, 1, 1, 11, 39, 113, 139, 165, 347, 595,)),
    (11, 265, (1, 3, 7, 11, 9, 17, 101, 13, 81, 325, 1733,)),
    (11, 266, (1, 3, 1, 1, 21, 43, 115, 9, 113, 907, 645,)),
    (11, 276, (1, 1, 7, 3, 9, 25, 117, 197, 159, 471, 475,)),
    (11, 292, (1, 3, 1, 9, 11, 21, 57, 207, 485, 613, 1661,)),
    (11, 304, (1, 1, 7, 7, 27, 55, 49, 223, 89, 85, 1523,)),
    (11, 310, (1, 1, 5, 3, 19, 41, 45, 51, 447, 299, 1355,)),
    (11, 316, (1, 3, 1, 13, 1, 33, 117, 143, 313, 187, 1073,)),
    (11, 319, (1, 1, 7, 7, 5, 11, 65, 97, 377, 377, 1501,)),
    (11, 322, (1, 3, 1, 1, 21, 35, 95, 65, 99, 23, 1239,)),
    (11, 328, (1, 1, 5, 9, 3, 37, 95, 167, 115, 425, 867,)),
    (11, 334, (1, 3, 3, 13, 1, 37, 27, 189, 81, 679, 773,)),
    (11, 339, (1, 1, 3, 11, 1, 61, 99, 233, 429, 969, 49,)),
    (11, 341, (1, 1, 1, 7, 25, 63, 99, 165, 245, 793, 1143,)),
    (11, 345, (1, 1, 5, 11, 11, 43, 55, 65, 71, 283, 273,)),
    (11, 346, (1, 1, 5, 5, 9, 3, 101, 251, 355, 379, 1611,)),
    (11, 362, (1, 1, 1, 15, 21, 63, 85, 99, 49, 749, 1335,)),
    (11, 367, (1, 1, 5, 13, 27, 9, 121, 43, 255, 715, 289,)),
    (11, 372, (1, 3, 1, 5, 27, 19, 17, 223, 77, 571, 1415,)),
    (11, 375, (1, 1, 5, 3, 13, 59, 125, 251, 195, 551, 1737,)),
    (11, 376, (1, 3, 3, 15, 13, 27, 49, 105, 389, 971, 755,)),
    (11, 381, (1, 3, 5, 15, 23, 43, 35, 107, 447, 763, 253,)),
    (11, 385, (1, 3, 5, 11, 21, 3, 17, 39, 497, 407, 611,)),
    (11, 388, (1, 1, 7, 13, 15, 31, 113, 17, 23, 507, 1995,)),
    (11, 392, (1, 1, 7, 15, 3, 15, 31, 153, 423, 79, 503,)),
    (11, 409, (1, 1, 7, 9, 19, 25, 23, 171, 505, 923, 1989,)),
    (11, 415, (1, 1, 5, 9, 21, 27, 121, 223, 133, 87, 697,)),
    (11, 416, (1, 1, 5, 5, 9, 19, 107, 99, 319, 765, 1461,)),
    (11, 421, (1, 1, 3, 3, 19, 25, 3, 101, 171, 729, 187,)),
    (11, 428, (1, 1, 3, 1, 13, 23, 85, 93, 291, 209, 37,)),
    (11, 431, (1, 1, 1, 15, 25, 25, 77, 253, 333, 947, 1073,)),
    (11, 434, (1, 1, 3, 9, 17, 29, 55, 47, 255, 305, 2037,)),
    (11, 439, (1, 3, 3, 9, 29, 63, 9, 103, 489, 939, 1523,)),
    (11, 446, (1, 3, 7, 15, 7, 31, 89, 175, 369, 339, 595,)),
    (11, 451, (1, 3, 7, 13, 25, 5, 71, 207, 251, 367, 665,)),
    (11, 453, (1, 3, 3, 3, 21, 25, 75, 35, 31, 321, 1603,)),
    (11, 457, (1, 1, 1, 9, 11, 1, 65, 5, 11, 329, 535,)),
    (11, 458, (1, 1, 5, 3, 19, 13, 17, 43, 379, 485, 383,)),
    (11, 471, (1, 3, 5, 13, 13, 9, 85, 147, 489, 787, 1133,)),
    (11, 475, (1, 3, 1, 1, 5, 51, 37, 129, 195, 297, 1783,)),
    (11, 478, (1, 1, 3, 15, 19, 57, 59, 181, 455, 697, 2033,)),
    (11, 484, (1, 3, 7, 1, 27, 9, 65, 145, 325, 189, 201,)),
    (11, 493, (1, 3, 1, 15, 31, 23, 19, 5, 485, 581, 539,)),
    (11, 494, (1, 1, 7, 13, 11, 15, 65, 83, 185, 847, 831,)),
    (11, 499, (1, 3, 5, 7, 7, 55, 73, 15, 303, 511, 1905,)),
    (11, 502, (1, 3, 5, 9, 7, 21, 45, 15, 397, 385, 597,)),
    (11, 517, (1, 3, 7, 3, 23, 13, 73, 221, 511, 883, 1265,)),
    (11, 518, (1, 1, 3, 11, 1, 51, 73, 185, 33, 975, 1441,)),
    (11, 524, (1, 3, 3, 9, 19, 59, 21, 39, 339, 37, 143,)),
    (11, 527, (1, 1, 7, 1, 31, 33, 19, 167, 117, 635, 639,)),
    (11, 555, (1, 1, 1, 3, 5, 13, 59, 83, 355, 349, 1967,)),
    (11, 560, (1, 1, 1, 5, 19, 3, 53, 133, 97, 863, 983,)),
)


def _sobol_direction_vectors(width: int) -> list[list[int]]:
    if width < 1 or width > len(_DIRECTION_ROWS) + 1:
        raise ValueError(f"embedded Sobol directions support widths 1..{len(_DIRECTION_ROWS) + 1}")

    directions = [[1 << (SOBOL_BITS - bit) for bit in range(1, SOBOL_BITS + 1)]]
    for degree, coefficient, initial in _DIRECTION_ROWS[: width - 1]:
        values = [0] * SOBOL_BITS
        for bit in range(1, degree + 1):
            values[bit - 1] = initial[bit - 1] << (SOBOL_BITS - bit)
        for bit in range(degree + 1, SOBOL_BITS + 1):
            value = values[bit - degree - 1] ^ (values[bit - degree - 1] >> degree)
            for k in range(1, degree):
                if (coefficient >> (degree - 1 - k)) & 1:
                    value ^= values[bit - k - 1]
            values[bit - 1] = value
        directions.append(values)
    return directions


def _fixed_antithetic_sobol(width: int) -> fnp.ndarray:
    directions = fnp.array(_sobol_direction_vectors(width), dtype=fnp.uint32)
    indices = fnp.arange(HALF_SAMPLES, dtype=fnp.uint32)
    gray = fnp.bitwise_xor(indices, fnp.right_shift(indices, 1))
    integers = fnp.zeros((HALF_SAMPLES, width), dtype=fnp.uint32)
    for bit in range((HALF_SAMPLES - 1).bit_length()):
        mask = fnp.bitwise_and(fnp.right_shift(gray, bit), 1)
        contribution = mask[:, None] * directions[:, bit][None, :]
        integers = fnp.bitwise_xor(integers, contribution)

    scramble = fnp.array(
        [
            random.Random(
                FIXED_POINTSET_SEED + SOBOL_SEED_OFFSET + 104729 * dimension
            ).getrandbits(SOBOL_BITS)
            for dimension in range(width)
        ],
        dtype=fnp.uint32,
    )
    shifted = fnp.bitwise_xor(integers, scramble[None, :])
    uniforms = fnp.clip(
        (shifted.astype(fnp.float64) + 0.5) / SOBOL_SCALE,
        1e-7,
        1.0 - 1e-7,
    )
    base = flops.stats.norm.ppf(uniforms).astype(fnp.float32)
    return fnp.concatenate((base, -base), axis=0)


def _gaussian_classifier(mlp: MLP) -> tuple[fnp.ndarray, tuple[fnp.ndarray, ...]]:
    width = mlp.width
    mu = fnp.zeros(width)
    cov = fnp.eye(width)
    log_scale = 0.0
    rows = []
    tail_alphas = []

    for layer_index, w in enumerate(mlp.weights):
        cov_diag = fnp.diag(cov)
        max_var = float(fnp.max(cov_diag))
        if max_var > _COV_RESCALE_THRESHOLD:
            scale = float(fnp.sqrt(max_var))
            mu = mu / scale
            cov = cov / (scale * scale)
            log_scale += float(fnp.log(scale))

        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sigma_pre = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        if layer_index >= mlp.depth - len(STRUCTURAL_THRESHOLDS):
            tail_alphas.append(alpha)

        phi_alpha = flops.stats.norm.pdf(alpha)
        Phi_alpha = flops.stats.norm.cdf(alpha)
        mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha
        ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
        var_post = fnp.maximum(ez2 - mu * mu, 0.0)

        sigma64 = fnp.asarray(sigma_pre, dtype=fnp.float64)
        gain64 = fnp.where(
            sigma64 > 1e-12,
            fnp.asarray(Phi_alpha, dtype=fnp.float64),
            0.0,
        )
        gain = fnp.array(gain64.astype(fnp.float32))
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        rows.append(mu * float(fnp.exp(log_scale)))

    prediction = fnp.multiply(fnp.stack(rows, axis=0), K2_COEFFICIENT)
    return prediction, tuple(tail_alphas)


def _classes(
    alpha: fnp.ndarray, threshold: float
) -> tuple[list[int], list[int], list[int]]:
    active = []
    dead = []
    uncertain = []
    for index, value in enumerate(alpha.tolist()):
        if value >= threshold:
            active.append(index)
        elif value <= -threshold:
            dead.append(index)
        else:
            uncertain.append(index)
    return active, dead, uncertain


def _sum_parts(parts: list[fnp.ndarray], width: int) -> fnp.ndarray:
    if not parts:
        return fnp.zeros(width, dtype=fnp.float32)
    result = parts[0]
    for part in parts[1:]:
        result = fnp.add(result, part)
    return result


def _structural_prediction(
    mlp: MLP,
    samples: fnp.ndarray,
    tail_alphas: tuple[fnp.ndarray, ...],
    identity: fnp.ndarray,
) -> fnp.ndarray:
    x = samples
    rows = []
    structural_depth = len(STRUCTURAL_THRESHOLDS)
    start = mlp.depth - structural_depth
    for w in mlp.weights[:start]:
        x = fnp.maximum(x @ w, 0.0)
        rows.append(fnp.mean(x, axis=0))

    previous_alive = None
    for offset, layer_index in enumerate(range(start, mlp.depth - 2)):
        active, _, uncertain = _classes(
            tail_alphas[offset], STRUCTURAL_THRESHOLDS[offset]
        )
        alive = active + uncertain
        if not alive:
            return fnp.stack(rows + [fnp.zeros(mlp.width)] * (mlp.depth - len(rows)))
        w = mlp.weights[layer_index]
        w_input = w if previous_alive is None else w[previous_alive, :]
        z_alive = x @ w_input[:, alive]
        parts = []
        if active:
            parts.append(z_alive[:, : len(active)])
        if uncertain:
            parts.append(fnp.maximum(z_alive[:, len(active) :], 0.0))
        x = parts[0] if len(parts) == 1 else fnp.concatenate(parts, axis=1)
        rows.append(fnp.mean(x, axis=0) @ identity[alive, :])
        previous_alive = alive

    w1 = mlp.weights[-2]
    w2 = mlp.weights[-1]
    w1_input = w1 if previous_alive is None else w1[previous_alive, :]
    active1, _, uncertain1 = _classes(
        tail_alphas[-2], STRUCTURAL_THRESHOLDS[-2]
    )
    active2, _, uncertain2 = _classes(
        tail_alphas[-1], STRUCTURAL_THRESHOLDS[-1]
    )

    mean_x = fnp.mean(x, axis=0)
    mean1_parts = []
    h1_uncertain = None
    if active1:
        active_mean1 = mean_x @ w1_input[:, active1]
        mean1_parts.append(active_mean1 @ identity[active1, :])
    if uncertain1:
        h1_uncertain = fnp.maximum(x @ w1_input[:, uncertain1], 0.0)
        uncertain_mean1 = fnp.mean(h1_uncertain, axis=0)
        mean1_parts.append(uncertain_mean1 @ identity[uncertain1, :])
    mean1 = _sum_parts(mean1_parts, mlp.width)

    mean2_parts = []
    if active2:
        active_mean2 = mean1 @ w2[:, active2]
        mean2_parts.append(active_mean2 @ identity[active2, :])
    if uncertain2:
        z2_uncertain = None
        if active1:
            active_to_uncertain = w2[active1, :][:, uncertain2]
            collapsed = w1_input[:, active1] @ active_to_uncertain
            z2_uncertain = x @ collapsed
        if uncertain1:
            uncertain_to_uncertain = w2[uncertain1, :][:, uncertain2]
            contribution = h1_uncertain @ uncertain_to_uncertain
            z2_uncertain = (
                contribution
                if z2_uncertain is None
                else fnp.add(z2_uncertain, contribution)
            )
        if z2_uncertain is not None:
            uncertain_mean2 = fnp.mean(fnp.maximum(z2_uncertain, 0.0), axis=0)
            mean2_parts.append(uncertain_mean2 @ identity[uncertain2, :])
    mean2 = _sum_parts(mean2_parts, mlp.width)

    return fnp.stack(rows + [mean1, mean2], axis=0)


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._samples = None
        self._identity = None
        self._width = None

    def setup(self, ctx: SetupContext) -> None:
        self._width = ctx.width
        submission_dir = (
            Path(ctx.submission_dir)
            if ctx.submission_dir is not None
            else Path(__file__).resolve().parent
        )
        payload = fnp.load(str(submission_dir / SAMPLES_FILE))
        stored_samples = payload["samples"]
        stored_identity = payload["identity"]
        if (
            stored_samples.shape != (N_SAMPLES, 256)
            or stored_identity.shape != (256, 256)
            or ctx.width > 256
        ):
            raise ValueError(
                f"invalid asset shapes {stored_samples.shape}, "
                f"{stored_identity.shape} for requested width {ctx.width}"
            )
        self._samples = stored_samples[:, : ctx.width]
        self._identity = stored_identity[: ctx.width, : ctx.width]

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        if self._samples is None or self._identity is None or self._width != mlp.width:
            raise RuntimeError("Estimator.setup() must run with the challenge width")

        k2, tail_alphas = _gaussian_classifier(mlp)
        if mlp.depth < len(STRUCTURAL_THRESHOLDS):
            return k2

        structural = _structural_prediction(
            mlp,
            self._samples,
            tail_alphas,
            self._identity,
        )
        return fnp.add(
            fnp.multiply(k2, 1.0 - LAMBDA),
            fnp.multiply(structural, LAMBDA),
        )
