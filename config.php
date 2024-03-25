<?php
define('DB_SERVER','localhost');
define('DB_USERNAME','root');
define('DB_PASSWORD','Prashant@123');
define('DB_NAME','plantify');

$conn = mysqli_connect(DB_SERVER,DB_USERNAME,DB_PASSWORD,DB_NAME);

if($conn->connect_errno ) {
    printf("Connect failed: %s<br />", $conn->connect_error);
    exit();
    }
// if($conn==false){
//     dir('Error: Cannot connect');
// }

?>