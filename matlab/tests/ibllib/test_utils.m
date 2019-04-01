function tests = test_time
tests = functiontests(localfunctions);
end

%% Test Functions
function test_aggregate(testCase)
% Test specific code
student.age   = [11 11 12 12 13 13 13 11 10 12 13]';
student.grade = [10  9  8  7  6  5  4  3  2  1  0]';
student.toto  = [ 1  1  2  2  1  2  1  2  1  2  1]';

% test the computation
[grad, age, ind] = aggregate(student.grade,student.age, @mean);
assert(all(age==[10 11 12 13]'))
assert(all(  abs(grad-[2;7.33333333333333;5.33333333333333;3.75]) < 0.001 ))

% test automatic transpose
[grad2, age2, ind2] = aggregate(student.grade,student.age', @mean);
assert(all(age2==age))
assert(all(grad2==grad))

% test matrix for the mapper
[grad, agetoto] = aggregate(student.grade,[ student.age student.toto], @mean);
assert(size(agetoto,2) == 2)

% test matrix for the feature
[grad, agetoto] = aggregate([student.grade student.age ] ,[ student.age], @mean);
assert(size(grad,2) == 2)

% test matrices for both
[grad, agetoto] = aggregate([student.grade student.age ] ,[ student.age student.toto], @mean);
assert(size(grad, 2) == 2)
assert(size(agetoto, 2) == 2)

end
