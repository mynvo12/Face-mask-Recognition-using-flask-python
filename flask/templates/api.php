<?php
$data = json_decode(file_get_contents('php://input'), true);

$curl = curl_init();

curl_setopt_array($curl, array(
  CURLOPT_URL => 'http://103.166.183.177:1234/api/v1/scan-qr',
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_ENCODING => '',
  CURLOPT_MAXREDIRS => 10,
  CURLOPT_TIMEOUT => 0,
  CURLOPT_FOLLOWLOCATION => true,
  CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
  CURLOPT_CUSTOMREQUEST => 'POST',
  CURLOPT_SSL_VERIFYPEER => false,
  CURLOPT_POSTFIELDS =>'{
    "phoneOtp": "0353975141",
    "qrCode": "'.$data['qrCode'].'"
}',
  CURLOPT_HTTPHEADER => array(
    'Content-Type: application/json'
  ),
));

$response = curl_exec($curl);

curl_close($curl);
/*
if($data['qrCode'] == "user|351981166639051|b722c8d9"){
  $arr = array (
    'status' => 0,
    'message' => 'Thành công',
    'data' =>
    array (
      'phone' => '0908 024 ***',
      'name' => 'NGUYEN VAN ***',
      'certification' => 'Bình thường (A+)',
      'health' => 'Bình thường (Khai báo y tế lần cuối vào ngày 21/09/2021 22:31:10)',
      'injection' => 'Đã tiêm 2 mũi Vắc-xin phòng COVID-19 và ngày gần nhất là 26/08/2021. Vẫn còn thời gian hiệu lực cho tới ngày 26/08/2022',
      'covidTest' => 'Chưa có thông tin',
    ),
  );
}else if($data['qrCode'] == "user|351981166639051|b722c8d8"){
  $arr = array (
    'status' => 0,
    'message' => 'Thành công',
    'data' =>
    array (
      'phone' => '0368 537 ***',
      'name' => 'TRAN THI ***',
      'certification' => 'Bình thường (A)',
      'health' => 'Bình thường (Khai báo y tế lần cuối vào ngày 21/09/2021 22:31:10)',
      'injection' => 'Đã tiêm 2 mũi Vắc-xin phòng COVID-19 và ngày gần nhất là 26/08/2021. Vẫn còn thời gian hiệu lực cho tới ngày 26/08/2022',
      'covidTest' => 'Chưa có thông tin',
    ),
  );
}else if($data['qrCode'] == "user|351981166639051|b722c8d7"){
  $arr = array (
    'status' => 0,
    'message' => 'Thành công',
    'data' =>
    array (
      'phone' => '0371 224 ***',
      'name' => 'HUYNH QUOC ***',
      'certification' => 'Bình thường (B)',
      'health' => 'Bình thường (Khai báo y tế lần cuối vào ngày 21/09/2021 22:31:10)',
      'injection' => 'Đã tiêm 1 mũi Vắc-xin phòng COVID-19 và ngày gần nhất là 26/08/2021. Vẫn còn thời gian hiệu lực cho tới ngày 26/08/2022',
      'covidTest' => 'Chưa có thông tin',
    ),
  );
}else if($data['qrCode'] == "user|351981166639051|b722c8d6"){
  $arr = array (
    'status' => 0,
    'message' => 'Thành công',
    'data' =>
    array (
      'phone' => '0949 024 ***',
      'name' => 'DUONG THANH ***',
      'certification' => 'Bình thường (C)',
      'health' => 'Chưa khai báo y tế',
      'injection' => 'Chưa có thông tin',
      'covidTest' => 'Chưa có thông tin',
    ),
  );
}else if($data['qrCode'] == "user|351981166639051|b722c8d5"){
  $arr = array (
    'status' => 0,
    'message' => 'Thành công',
    'data' =>
    array (
      'phone' => '0935 148 ***',
      'name' => 'TRAN XUAN ***',
      'certification' => 'Bình thường (D)',
      'health' => 'Nhóm nguy cơ lây nhiễm',
      'injection' => 'Chưa có thông tin',
      'covidTest' => 'Chưa có thông tin',
    ),
  );
}else if($data['qrCode'] == "user|351981166639051|b722c8d4"){
  $arr = array (
    'status' => 0,
    'message' => 'Thành công',
    'data' =>
    array (
      'phone' => '0968 145 ***',
      'name' => 'LE THI ANH ***',
      'certification' => 'Bình thường (Z)',
      'health' => 'Dương tính Covid',
      'injection' => 'Chưa có thông tin',
      'covidTest' => 'Chưa có thông tin',
    ),
  );
}
echo json_encode($arr);*/
echo $response;
 ?>
