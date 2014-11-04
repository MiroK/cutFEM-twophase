size = 1;

Point(1) = {-1, -1, 0, size};
Point(2) = {1, -1, 0, size};
Point(3) = {1, 1, 0, size};
Point(4) = {-1, 1, 0, size};
Line(1) = {4, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};
Line Loop(5) = {4, 1, 2, 3};
Plane Surface(6) = {5};
