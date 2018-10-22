function tests = test_time
tests = functiontests(localfunctions);
end

function test_json2serial(testCase)

expected = [ '2018-09-21T12:00:00:00000';'2018-10-03T14:16:13:92900'];
a = '2018-09-21T12:00:00';
b = '2018-10-03T14:16:13.929370';
dt = time.jsonrest2serial({a,b});
testCase.assertEqual(time.serial2iso8601(dt), expected);
end