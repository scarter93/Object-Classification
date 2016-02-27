#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <conio.h>
#include <windows.h>

using namespace cv;
using namespace std;

/* Helper class declaration and definition */
class Caltech101
{
public:
	Caltech101::Caltech101(string datasetPath, const int numTrainingImages, const int numTestImages)
	{
		successfullyLoaded = false;
		cout << "Loading Caltech 101 dataset" << endl;
		numImagesPerCategory = numTrainingImages + numTestImages;

		// load "Categories.txt"
		ifstream infile(datasetPath + "/" + "Categories.txt");
		cout << "\tChecking Categories.txt" << endl;
		if (!infile.is_open())
		{
			cout << "\t\tError: Cannot find Categories.txt in " << datasetPath << endl;
			return;
		}
		cout << "\t\tOK!" << endl;

		// Parse category names
		cout << "\tParsing category names" << endl;
		string catname;
		while (getline(infile, catname))
		{
			categoryNames.push_back(catname);
		}
		cout << "\t\tdone!" << endl;

		// set num categories
		int numCategories = (int)categoryNames.size();

		// initialize outputs size
		trainingImages = vector<vector<Mat>>(numCategories);
		trainingAnnotations = vector<vector<Rect>>(numCategories);
		testImages = vector<vector<Mat>>(numCategories);
		testAnnotations = vector<vector<Rect>>(numCategories);

		// generate training and testing indices
		randomShuffle();

		// Load data
		cout << "\tLoading images and annotation files" << endl;
		string imgDir = datasetPath + "/" + "Images";
		string annotationDir = datasetPath + "/" + "Annotations";
		for (int catIdx = 0; catIdx < categoryNames.size(); catIdx++)
			//for (int catIdx = 0; catIdx < 1; catIdx++)
		{
			string imgCatDir = imgDir + "/" + categoryNames[catIdx];
			string annotationCatDir = annotationDir + "/" + categoryNames[catIdx];
			for (int fileIdx = 0; fileIdx < numImagesPerCategory; fileIdx++)
			{
				// use shuffled training and testing indices
				int shuffledFileIdx = indices[fileIdx];
				// generate file names
				stringstream imgFilename, annotationFilename;
				imgFilename << "image_" << setfill('0') << setw(4) << shuffledFileIdx << ".jpg";
				annotationFilename << "annotation_" << setfill('0') << setw(4) << shuffledFileIdx << ".txt";

				// Load image
				string imgAddress = imgCatDir + '/' + imgFilename.str();
				Mat img = imread(imgAddress, CV_LOAD_IMAGE_COLOR);
				// check image data
				if (!img.data)
				{
					cout << "\t\tError loading image in " << imgAddress << endl;
					return;
				}

				// Load annotation
				string annotationAddress = annotationCatDir + '/' + annotationFilename.str();
				ifstream annotationIFstream(annotationAddress);
				// Checking annotation file
				if (!annotationIFstream.is_open())
				{
					cout << "\t\tError: Error loading annotation in " << annotationAddress << endl;
					return;
				}
				int tl_col, tl_row, width, height;
				Rect annotRect;
				while (annotationIFstream >> tl_col >> tl_row >> width >> height)
				{
					annotRect = Rect(tl_col - 1, tl_row - 1, width, height);
				}

				// Split training and testing data
				if (fileIdx < numTrainingImages)
				{
					// Training data
					trainingImages[catIdx].push_back(img);
					trainingAnnotations[catIdx].push_back(annotRect);
				}
				else
				{
					// Testing data
					testImages[catIdx].push_back(img);
					testAnnotations[catIdx].push_back(annotRect);
				}
			}
		}
		cout << "\t\tdone!" << endl;
		successfullyLoaded = true;
		cout << "Dataset successfully loaded: " << numCategories << " categories, " << numImagesPerCategory << " images per category" << endl << endl;
	}

	bool isSuccessfullyLoaded() { return successfullyLoaded; }

	void dispTrainingImage(int categoryIdx, int imageIdx)
	{
		Mat image = trainingImages[categoryIdx][imageIdx];
		Rect annotation = trainingAnnotations[categoryIdx][imageIdx];
		rectangle(image, annotation, Scalar(255, 0, 255), 2);
		imshow("Annotated training image", image);
		waitKey(0);
		destroyWindow("Annotated training image");
	}

	void dispTestImage(int categoryIdx, int imageIdx)
	{
		Mat image = testImages[categoryIdx][imageIdx];
		Rect annotation = testAnnotations[categoryIdx][imageIdx];
		rectangle(image, annotation, Scalar(255, 0, 255), 2);
		imshow("Annotated test image", image);
		waitKey(0);
		destroyWindow("Annotated test image");
	}

	vector<string> categoryNames;
	vector<vector<Mat>> trainingImages;
	vector<vector<Rect>> trainingAnnotations;
	vector<vector<Mat>> testImages;
	vector<vector<Rect>> testAnnotations;

private:
	bool successfullyLoaded;
	int numImagesPerCategory;
	vector<int> indices;
	void randomShuffle()
	{
		// set init values
		for (int i = 1; i <= numImagesPerCategory; i++) indices.push_back(i);

		// permute using built-in random generator
		random_shuffle(indices.begin(), indices.end());
	}
};

/* Function prototypes */
void Train(const Caltech101 &Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords,vector<vector<vector<KeyPoint>>> &results, Mat &D, Ptr<DescriptorExtractor> descriptor_extractor);
void Test(const Caltech101 &Dataset, const Mat codeBook, const vector<vector<Mat>> &imageDescriptors);
static bool LiesInside(Rect rectangle, KeyPoint point);

void keypoints_train(const Caltech101 &Dataset, vector<vector<vector<KeyPoint>>> &results, Mat &D, Ptr<DescriptorExtractor> descriptor_extractor);

std::ofstream myfile;

int main(void)
{
	/* Initialize OpenCV nonfree module */
	initModule_nonfree();

	/* Put the full path of the Caltech 101 folder here */
	const string datasetPath = "C:\\Users\\scarte9\\Caltech 101";

	/* Set the number of training and testing images per category */
	const int numTrainingData = 40;
	const int numTestingData = 2;

	/* Set the number of codewords*/
	const int numCodewords[13] = {10,20,50,100,200,300,400,500,600,700,800,900,1000};

	/* Load the dataset by instantiating the helper class */
	Caltech101 Dataset(datasetPath, numTrainingData, numTestingData);

	/* Terminate if dataset is not successfull loaded */
	if (!Dataset.isSuccessfullyLoaded())
	{
		cout << "An error occurred, press Enter to exit" << endl;
		getchar();
		return -1;
	}

	/* Variable definition */
	Mat codeBook;
	vector<vector<Mat>> imageDescriptors;
	vector<vector<Mat>> categoryDescriptor;

	Ptr<DescriptorExtractor> descriptor_extract = DescriptorExtractor::create("SIFT");

	Mat D;
	vector<vector<vector<KeyPoint>>> results;

		
	keypoints_train(Dataset, results, D, descriptor_extract);
<<<<<<< HEAD
	myfile.open("results_test_noremove.txt");
=======
	myfile.open("results_train.txt");
>>>>>>> training-1
	/* Training */
	for(int i = 0; i < 13; i++) {
		Train(Dataset, codeBook, imageDescriptors, numCodewords[i], results, D, descriptor_extract);

	/* Testing */
		Test(Dataset, codeBook, imageDescriptors);
	}

<<<<<<< HEAD
=======
	myfile.close();
>>>>>>> training-1
	std::system("Pause");

	return 1;
}

void keypoints_train(const Caltech101 &Dataset,vector<vector<vector<KeyPoint>>> &results, Mat &D, Ptr<DescriptorExtractor> descriptor_extract){

	Ptr<FeatureDetector> feature_detector = FeatureDetector::create("SIFT");
	vector<KeyPoint> store_kp;

	int k = 0;
	int images = Dataset.trainingImages[0].size();
	int category = Dataset.trainingImages.size();
	Mat I;
	Mat current_d;

	results.resize(Dataset.trainingImages.size());
	cout << "Generating Keypoints 3" << endl;
	for (int i = 0; i < category; i++) {
		results[i].resize(Dataset.trainingImages[i].size());
		for (int j = 0; j < images; j++) { 

			I = Dataset.trainingImages[i][j];
			feature_detector->detect(I, store_kp);

			Rect to_remove = Dataset.trainingAnnotations[i][j];

			store_kp.erase(
				std::remove_if(
					store_kp.begin(), store_kp.end(),
					[&to_remove](KeyPoint k) { return !to_remove.contains(k.pt); }),
					store_kp.end()
				);
			
			results[i][j] = store_kp;
			descriptor_extract->compute(I, store_kp, current_d);
			D.push_back(current_d);

			/*Mat image_out;
			drawKeypoints(I, store_kp, image_out, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			rectangle(image_out, Dataset.trainingAnnotations[i][j], Scalar(255, 255, 255));
			std:ostringstream os;
			os << "sift_keypoints_filtered" << i << ".jpg";
			imwrite(os.str(), image_out);*/

		}
	}
}

/* Train BoW */
void Train(const Caltech101 &Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords,vector<vector<vector<KeyPoint>>> &results, Mat &D, Ptr<DescriptorExtractor> descriptor_extract)
{
	int images = Dataset.trainingImages[0].size();
	int category = Dataset.trainingImages.size();
	Mat I;
	BOWKMeansTrainer trainer(numCodewords);
	trainer.add(D);
	codeBook = trainer.cluster();
	
	Ptr<DescriptorMatcher> desc_matcher = DescriptorMatcher::create("BruteForce");
	Ptr<BOWImgDescriptorExtractor> BOW_extract = new ::BOWImgDescriptorExtractor(descriptor_extract, desc_matcher);

	BOW_extract->setVocabulary(codeBook);

	
	
	imageDescriptors.resize(Dataset.trainingImages.size());
	cout << "Creating Histograms" << endl;
	for (int i = 0; i < category; i++) {
		imageDescriptors[i].resize(Dataset.trainingImages[i].size());
		for (int j = 0; j < images; j++) {
			
			Mat BOW_obj;
			I = Dataset.trainingImages[i][j];
			vector<KeyPoint> &input = results[i][j];
			BOW_extract->compute2(I, input, BOW_obj);
			imageDescriptors[i][j] = BOW_obj;
		}
	}

	cout << "Training complete" << endl;

	//namedWindow("Keypoints");
	/*drawKeypoints(I, store_kp, image_out, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	rectangle(image_out, Dataset.trainingAnnotations[0][0],Scalar(255,255,255));
	imshow("Keypoints", image_out);
	waitKey(1);*/
	
	//std::system("Pause");
}


/* Test BoW */
void Test(const Caltech101 &Dataset, const Mat codeBook, const vector<vector<Mat>> &imageDescriptors)
{
	Ptr<FeatureDetector> feature_detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptor_extract = DescriptorExtractor::create("SIFT");

	Ptr<DescriptorMatcher> desc_matcher = DescriptorMatcher::create("BruteForce");
	Ptr<BOWImgDescriptorExtractor> BOW_extract = new ::BOWImgDescriptorExtractor(descriptor_extract, desc_matcher);

	BOW_extract->setVocabulary(codeBook);

	int category = Dataset.trainingImages.size();
	int image = Dataset.trainingImages[0].size();
	vector<vector<vector<KeyPoint>>> results;
	vector<KeyPoint> store_kp;

	Mat I;
	Mat current_d;
	Mat BOW_obj;

	results.resize(Dataset.trainingImages.size());
	cout << "Generating Keypoints" << endl;
	for (int i = 0; i < category; i++) {
		results[i].resize(Dataset.trainingImages[i].size());
		for (int j = 0; j < image; j++) {

			I = Dataset.trainingImages[i][j];
			feature_detector->detect(I, store_kp);

			Rect to_remove = Dataset.trainingAnnotations[i][j];

			store_kp.erase(
				std::remove_if(
					store_kp.begin(), store_kp.end(),
					[&to_remove](KeyPoint k) { return !to_remove.contains(k.pt); }),
				store_kp.end()
				);

			results[i][j] = store_kp;

			/*Mat image_out;
			drawKeypoints(I, store_kp, image_out, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			rectangle(image_out, Dataset.trainingAnnotations[i][j], Scalar(255, 255, 255));
			std:ostringstream os;
			os << "sift_keypoints_filtered" << i << ".jpg";
			imwrite(os.str(), image_out);*/

		}
	}


	cout << "Creating and comparing histograms" << endl;
	int correct = 0;
	int total = 0;
	int cat = 0;

	int category_t = Dataset.trainingImages.size();
	int images_t = Dataset.trainingImages[0].size();
	string label;
	for (int i = 0; i < category; i++) {
		for (int j = 0; j < image; j++) {

			I = Dataset.trainingImages[i][j];
			BOW_extract->compute2(I, results[i][j], BOW_obj);

			/*cout << "Size of test Imag: " << BOW_obj.size() << endl;
			cout << "Size of descriptor: " << imageDescriptors[0][0].size() << endl;*/

			double min = DBL_MAX;

			for (int k = 0; k < category_t; k++) {
				for (int l = 0; l < images_t; l++) {
					double value = norm(BOW_obj, imageDescriptors[k][l]);
					if (value < min) {
						min = value;
						label = Dataset.categoryNames[k];
						cat = k;
					}
				}
			}
			total++;
			if (cat == i)
				correct++;
			//imshow(label, I);
			waitKey(1);

			/*std:ostringstream os;
			os << "test_image" << i << "_" << j << "_result_" << label << ".jpg";
			imwrite(os.str(), I);*/

		}
	}
	float rec_rate = ((float)correct) / ((float)total);
	cout << "Recognition Rate is: " << rec_rate << endl;

	myfile << "Recognition Rate is: " << rec_rate << "\n";

	//std::system("pause");
}