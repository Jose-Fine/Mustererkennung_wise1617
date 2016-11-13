
function aufgabe1(class1, class2, alpha=1e-2)
	samples1 = csvread(strcat("train.", num2str(class1)));
	samples2 = csvread(strcat("train.", num2str(class2)));
	
	X = [ones(size(samples1,1),1), samples1; 
		ones(size(samples2,1),1), samples2];
	y = [ones(size(samples1,1),1);
		-ones(size(samples2,1),1)];
	
	beta = (X'*X + alpha*eye(size(X,2)))^-1 * X' * y;
	
	testData = dlmread("zip.test", " ");
	testData1 = testData(testData(:,1) == class1,:);
	testData2 = testData(testData(:,1) == class2,:);
	
	testData1(:,1) = 1;
	testData2(:,1) = 1;
	
	classifiedAs1 = testData1 * beta;
	classifiedAs2 = testData2 * beta;
	printf("%d vs %d quality: %f and %f\n", class1, class2, 
		size(classifiedAs1(classifiedAs1 > 0), 1) / size(testData1, 1),
		size(classifiedAs2(classifiedAs2 < 0), 1) / size(testData2, 1))
end

aufgabe1(3,5)
aufgabe1(3,7)
aufgabe1(3,8)
aufgabe1(5,7)
aufgabe1(5,8)
aufgabe1(7,8)
