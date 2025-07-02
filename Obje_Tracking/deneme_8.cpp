#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#define NOMINMAX
#include <Windows.h>
#include <map>
#include <thread>
#include <mutex>
#include <limits>
#include <string.h>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>



std::map<std::string, std::vector<std::pair<cv::Scalar, cv::Scalar>>> color_ranges = { {"red",{{cv::Scalar(0,120,70),cv::Scalar(10,255,255)},{cv::Scalar(170,120,70),cv::Scalar(180,255,255)}}},
	{"blue",{{cv::Scalar(100,150,0),cv::Scalar(140,255,255)}}},
	{"green",{{cv::Scalar(35,100,100),cv::Scalar(85,255,255)}}},
	{"yellow",{{cv::Scalar(20,100,100),cv::Scalar(30,255,255)}}},
	{"black",{{cv::Scalar(0,0,0),cv::Scalar(180,255,50)}}} };

enum Mode { Color_Detection, OCR_Detection, Exit, None, Yorunge_Algilama, Manuel, Face_Detection };
Mode current_mode = None; // Baþlangýç modu
std::mutex mode_mutex; // Mod deðiþiklikleri için mutex

std::string selected_color = "";
std::mutex color_mutex;

std::vector<std::vector<cv::Point>> filtered_contour;

std::vector<cv::Point> detected_center;

std::vector<double> previous_velocityX, previous_velocityY;

double startTime, currentTime, deltaTime;

double errorX = 0;
double nextX_arti_pid = 0;

double errorY = 0;
double nextY_arti_pid = 0;

bool pid_hesap = false;

HANDLE hSerial = INVALID_HANDLE_VALUE;



void detectSelectColor(cv::Mat frame, std::string selected_color) {

	if (color_ranges.find(selected_color) == color_ranges.end()) {
		std::cout << "Bilinmeyen Tespit Edilemeyen Renk !";
		return;
	}
	filtered_contour.clear();
	cv::Mat hsv, mask, blured;
	cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
	cv::GaussianBlur(hsv, blured, cv::Size(5, 5), 0);
	mask = cv::Mat::zeros(frame.size(), CV_8U);

	for (const auto& range : color_ranges[selected_color]) {
		cv::Mat temp_mask;
		cv::inRange(hsv, range.first, range.second, temp_mask);
		mask |= temp_mask;
	}
	cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	for (const auto& cnt : contours) {
		double area = cv::contourArea(cnt);
		if (area > 500) {
			filtered_contour.push_back(cnt);
		}
	}
	//çoklu nesne için moment hesaplamasasý
	int index = 1;
	for (const auto& cnt : filtered_contour) {
		cv::Moments M = cv::moments(cnt);
		if (M.m00 != 0) {
			double object_centerX = static_cast<int>(M.m10 / M.m00);
			double object_centerY = static_cast<int>(M.m01 / M.m00);
			cv::Point center(object_centerX, object_centerY);
			detected_center.push_back(center);

			//std::cout << index << ". Nesne" << "   " << "Merkezi: (" << center.x << ", " << center.y << ")" << std::endl;
			index++;
			cv::circle(frame, center, 5, cv::Scalar(0, 0, 255), -1);
		}
	}
	if (detected_center.empty()) {
		detected_center.push_back(cv::Point(0, 0));
	}
	cv::drawContours(frame, filtered_contour, -1, cv::Scalar(0, 255, 0), 1);
	cv::imshow("renk", frame);
}

void inputThreat() {
	while (true) {
		std::string input;
		std::cout << "(red, blue, green, yellow, black)---(ocr)---(yorunge)---(manuel)---(face_detection)---(exit) yaz: ";
		std::cin >> input;

		std::lock_guard<std::mutex> lock(mode_mutex);


		if (input == "exit") {
			current_mode = Exit;
			break;
		}
		else if (input == "ocr") {
			current_mode = OCR_Detection;
			std::cout << "OCR moduna gecildi.\n";
		}
		else if (color_ranges.find(input) != color_ranges.end()) {
			current_mode = Color_Detection;
			selected_color = input;
			std::cout << "Renk moduna gecildi: " << selected_color << std::endl;
		}
		else if (input == "yorunge") {
			current_mode = Yorunge_Algilama;
			std::cout << "Yorunge Algilama moduna gecildi.\n";
		}
		else if (input == "manuel") {
			current_mode = Manuel;
			std::cout << "Manuel Algilama moduna gecildi.\n";
		}
		else if (input == "face_detection") {
			current_mode = Face_Detection;
			std::cout << "Yuz Algilama moduna gecildi.\n";
		}
		else {
			current_mode = None;
			std::cout << "Gecersiz giris. Lutfen tekrar deneyin.\n";
		}
	}
}

//ivme ve hýz kullanarak bir sonraki center_point tahmin etme 
void predictPosition() {

	currentTime = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
	deltaTime = currentTime - startTime;
	startTime = currentTime;
	//std::cout << "fps: " << deltaTime << " saniye" << std::endl;

	static std::vector<cv::Point> previous_detected_center; //bunu static yapmak kodu hýzlandýrdý static yap
	static double previous_velocityX = 0;
	static double previous_velocityY = 0;
	static double previous_nextX = 0;
	static double previous_nextY = 0;
	double nextX = 0;
	double nextY = 0;

	if (previous_detected_center.empty()) {
		previous_detected_center = detected_center;
		return;
	}

	int x = 0;
	pid_hesap = false;
	for (size_t i = 0;i < detected_center.size();i++) {
		if (i < previous_detected_center.size()) {

			//std::cout << "i:" << i << "now" << detected_center[i] << "--" << "previous" << previous_detected_center[i] << "--";
			std::cout << "Nesne Konumu :" << detected_center[i] << "------";
			double deltaX = static_cast<double>(detected_center[i].x - previous_detected_center[i].x);
			double deltaY = static_cast<double>(detected_center[i].y - previous_detected_center[i].y);

			double velocityX = deltaX / deltaTime;
			double velocityY = deltaY / deltaTime;

			double old_velocityX = previous_velocityX;
			double old_velocityY = previous_velocityY;
			previous_velocityY = velocityY;
			previous_velocityX = velocityX;

			double accelerationX = (velocityX - old_velocityX) / deltaTime;
			double accelerationY = (velocityY - old_velocityY) / deltaTime;

			nextX = detected_center[i].x + static_cast<double>(velocityX * deltaTime) + 0.5 * accelerationX * deltaTime * deltaTime;
			nextY = detected_center[i].y + static_cast<double>(velocityY * deltaTime) + 0.5 * accelerationY * deltaTime * deltaTime;

			errorX = detected_center[i].x - previous_nextX;//hata þimdi oldu gibi orta nokta hesaplýo sonra bir sonraki nokta tahmin yapýyo bir sonraki nokta - onceki tahmin = error
			errorY = detected_center[i].y - previous_nextY;
			nextX_arti_pid = nextX;
			nextY_arti_pid = nextY;
			//std::cout << "errorX = " << errorX << "-";
			//std::cout << "previous_nextX = " << previous_nextX << "-";


			pid_hesap = true;

		}
	}
	previous_detected_center = detected_center;
	previous_nextX = nextX;
	previous_nextY = nextY;
	detected_center.clear();
}

double calculatePid() { 

	double Kp = 0.08;
	double Kd = 0.015; 
	double Ki = 0.003;
	static double previous_errorX = 0;
	static double integral = 0;
	double pid_output = 0;
	double turev = 0;

	if (pid_hesap == true) {

		integral += errorX * deltaTime;
		turev = (errorX - previous_errorX) / deltaTime;
		pid_output = (Kp * errorX) + (Ki * integral) + (Kd * turev);

		//std::cout << "Kp=" << Kp << " " << "Kp*error=" << Kp * errorX << "-";
		//std::cout << "kd=" << Kd << " " << "turev=" << turev << "-";
		//std::cout << "ki=" << Ki << " " << "integral=" << integral << "-";
		//std::cout << "pid_output" << pid_output << std::endl;
	}
	else {
		integral = 0;
		pid_output = 0;
		previous_errorX = 0;
	}

	previous_errorX = errorX;
	return pid_output;
}

double calculatePid_Y() {

	double Kp = 0.08;
	double Kd = 0.015;
	double Ki = 0.003;
	static double previous_errorY = 0;
	static double integral_Y = 0;
	double pid_output_Y = 0;
	double turev_Y = 0;

	if (pid_hesap == true) {
		integral_Y += errorY * deltaTime;
		turev_Y = (errorY - previous_errorY) / deltaTime;
		pid_output_Y = (Kp * errorY) + (Ki * integral_Y) + (Kd * turev_Y);
	}
	else {
		integral_Y = 0;
		pid_output_Y = 0;
		previous_errorY = 0;
	}
	previous_errorY = errorY;
	return pid_output_Y;
}


double pid_uygulanmis_deger() {

	double pid_arti_tahmin = nextX_arti_pid + calculatePid();
	std::cout << "konum_tahmin = " << pid_arti_tahmin << std::endl;
	return pid_arti_tahmin;
}
double pid_uygulanmis_deger_Y() {
	double pid_arti_tahmin_Y = nextY_arti_pid + calculatePid_Y();
	std::cout << "konum_tahmin_Y = " << pid_arti_tahmin_Y << std::endl;
	return pid_arti_tahmin_Y;
}


//port ayarlarý
bool initSerialPort() {

	hSerial = CreateFile(L"\\\\.\\COM7", GENERIC_WRITE | GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);
	if (hSerial == INVALID_HANDLE_VALUE) {
		std::cerr << "COM7 açilamadi!" << std::endl;
		return false;
	}

	DCB dcbSerialParams = { 0 };
	dcbSerialParams.DCBlength = sizeof(dcbSerialParams);
	if (!GetCommState(hSerial, &dcbSerialParams)) {
		std::cerr << "Seri ayarlar alinamadi." << std::endl;
		CloseHandle(hSerial);
		return false;
	}

	dcbSerialParams.BaudRate = CBR_115200;
	dcbSerialParams.ByteSize = 8;
	dcbSerialParams.StopBits = ONESTOPBIT;
	dcbSerialParams.Parity = NOPARITY;

	if (!SetCommState(hSerial, &dcbSerialParams)) {
		std::cerr << "Seri ayarlar yapilamadi." << std::endl;
		CloseHandle(hSerial);
		return false;
	}

	COMMTIMEOUTS timeouts = { 0 };
	timeouts.WriteTotalTimeoutConstant = 10;
	SetCommTimeouts(hSerial, &timeouts);

	return true;
}
//veri gönderme fonksiyonu
bool sendDataToStm32(double value1, double value2) {

	if (hSerial == INVALID_HANDLE_VALUE) {
		std::cerr << "Seri port geçerli deðil!\n";
		return false;
	}

	double values[2] = { value1, value2 };
	DWORD bytes_written = 0;
	if (!WriteFile(hSerial, &values, sizeof(values), &bytes_written, NULL)) {
		std::cerr << "Veri gönderilemedi.\n";
		return false;
	}
	std::cout << "Gonderilen veriler: " << value1 << ", " << value2 << " (" << bytes_written << " byte)" << std::endl;
	return true;
}
//stm32 den veri okuma fonksiyonu
bool readDataToStm32() {
	if (hSerial == INVALID_HANDLE_VALUE) {
		std::cerr << "Seri port geçerli deðil!\n";
		return false;
	}
	char buffer[64];
	DWORD bytes_read;
	if (!ReadFile(hSerial, buffer, sizeof(buffer) - 1, &bytes_read, NULL)) {
		std::cerr << "Veri okunamadý.\n";
		return false;
	}
	if (bytes_read > 0) {
		buffer[bytes_read] = '\0';
		std::cout << "STM32 den gelen veri: " << buffer << std::endl;
	}
	else {
		std::cout << "STM32 den veri yok.\n";
	}
	return true;
}
//port kapama fonksiyonu
void closeSerialPort() {
	if (hSerial != INVALID_HANDLE_VALUE) {
		CloseHandle(hSerial);
		hSerial = INVALID_HANDLE_VALUE;
	}
}

//--------------------------------------------------------------------------------------------------

//TESSERACT ÝLE YAZI OKUMA KODU !!

bool zoom_butonu = false;
double zoom_butonu_deger = 1; //zoom_butonu basýlýnca artcak deger

bool capture_butonu = false; //butona basýlýnca kameradan fotoðrafý çekicek
bool islem_butonu = false; //butona basýlýrsa kameradan resim çeksin sonra resime  tesseract uygulasýn

std::string detectTextwithTesseract(cv::Mat& frame) {
	static cv::Mat captured_frame;
	cv::Mat zoomed;

	std::string outText = "";

	static int image_counter = 0;
	std::string klasor_yolu = "C:\\Users\\Halit\\Desktop\\tesseract_img\\";
	std::string dosya_adi = klasor_yolu + "captured_image" + std::to_string(image_counter++) + ".jpg";

	frame.copyTo(captured_frame); //frame kopyalanýyor

	//kameradan görüntün alýyosam
	if (!captured_frame.empty()) {

		//zoom buton deðerine sýnýrlama koymak lazým !
		if (zoom_butonu == true) {
			zoom_butonu_deger += 0.5;
			int w = static_cast<int>(captured_frame.cols / zoom_butonu_deger);
			int h = static_cast<int>(captured_frame.rows / zoom_butonu_deger);
			int x = static_cast<int>((captured_frame.cols - w) / 2);
			int y = static_cast<int>((captured_frame.rows - h) / 2);
			cv::Rect zoom_roi(static_cast<int>(x), static_cast<int>(y), static_cast<int>(w), static_cast<int>(h));
			zoomed = captured_frame(zoom_roi).clone();
			cv::resize(zoomed, zoomed, captured_frame.size());

			zoom_butonu = false;//bunu kaldir kontrol için 
		}
		else {
			zoomed = captured_frame.clone();
		}

		if (islem_butonu == true) {
			bool kaydetme_check = cv::imwrite(dosya_adi, zoomed); //zoomed resmini kaydetme iþlemi
			if (kaydetme_check) {
				std::cout << "Görüntü kaydedildi: " << dosya_adi << std::endl;

				cv::Mat ocr_image = cv::imread(dosya_adi);
				if (!ocr_image.empty()) {
					cv::Mat gray;
					cv::cvtColor(ocr_image, gray, cv::COLOR_BGR2GRAY);

					tesseract::TessBaseAPI tess;
					const char* tessdata_path = "C:/Program Files/Tesseract-OCR/tessdata";
					if (tess.Init(tessdata_path, "eng+tur") != 0) {
						std::cerr << "tesseract baslatilamadi !" << std::endl;
						return "";
					}
					tess.SetImage(gray.data, gray.cols, gray.rows, 1, gray.step);
					char* text = tess.GetUTF8Text();
					if (text) {
						outText = std::string(text);
						delete[] text;
					}
					tess.End();
				}
			}
			else {
				std::cerr << "Görüntü kaydedilemedi!" << std::endl;
			}
		}


		cv::Mat gray;
		cv::cvtColor(zoomed, gray, cv::COLOR_BGR2GRAY);
		cv::imshow("Gri Tonlama", gray);
		cv::imshow("Kamera Goruntusu", captured_frame);
	}
	cv::waitKey(1);
	return outText;
}


std::vector<std::pair<double, double>> yorunge_fonksiyonu() {



}



int main() {

	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "Kamera acilmadi \n";
		return -1;
	}

	//COM7 PORTU AÇMA KODU BUNU AKTÝFLEÞTÝRMEN LAZIM STM32 ÝLE HABERLEÞEBÝLMEK ÝÇÝN !
	//bu port garanti açýlmak zorunda böyle kalabilir yani bu koþula baðlý deðil!
	//þimdilik aktif etme çünkü stm32 kartýnýn com7 portuna baðlý olmasý gerek.

	if (!initSerialPort()) {
		return -1;
	}

	startTime = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();

	std::thread input_thread(inputThreat);
	cv::Mat frame;

	while (true) {
		cap >> frame;
		if (frame.empty()) break;
		Mode mode_copy;
		std::string color_copy;
		{
			//bu kýsýmda modlar için kullancaðýn girdiler falan kullanýcý tarafýndan giriliyor veya deðiþiyor ise 
			//tanýmlayýp sonradan ifler içinde veya baþka fonskyionlarýn elemaný olarak kullanabilirsin ornek color_copy
			//detectionSelectColor fonksiyonunda kullanýlýyor.
			std::lock_guard<std::mutex> lock(mode_mutex);
			mode_copy = current_mode; //current_mode deðiþkeni mode_copy elemaný ile kopyalanýyor kullanýrken mode_copy kullanýyoz.
			color_copy = selected_color;
		}

		if (mode_copy == Exit) break;
		if (mode_copy == None) continue;

		
		if (mode_copy == Color_Detection) {
			detectSelectColor(frame, color_copy);
			predictPosition();
			//burada calculatepid fonksiyonu 1 kere çalýþýyor sonra bunu pid_uygulanmýs_deger içinde bir kere daha çalýþtýrýyoruz hatalý bunu düzelt.
			calculatePid();
			pid_uygulanmis_deger();
			calculatePid_Y();
			pid_uygulanmis_deger_Y();
			sendDataToStm32(pid_uygulanmis_deger(), pid_uygulanmis_deger_Y());
			//readDataToStm32();

		}
		if (mode_copy == OCR_Detection) {
			islem_butonu = false;
			std::string text = detectTextwithTesseract(frame);
			if (!text.empty()) {
				std::cout << "OCR Okunan Metin: " << text << std::endl;
			}
		}
		
		if (mode_copy == Yorunge_Algilama) {
			islem_butonu = true;
			std::string text = detectTextwithTesseract(frame);
			if (!text.empty()) {
				std::cout << "OCR Okunan Metin: " << text << std::endl;
			}
			//continue;
		}
		//joystick ile sürme 
		if (mode_copy == Manuel) {
			continue;
		}
		//buraya baþka birþey eklenebilir!
		if (mode_copy == Face_Detection) {
			continue;
		}

		if (cv::waitKey(1) == 27) break;
	}
	input_thread.join();
	cap.release();
	cv::destroyAllWindows();
	closeSerialPort();
	return 0;
}
