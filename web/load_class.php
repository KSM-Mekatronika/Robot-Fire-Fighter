<?php
$servername = "localhost";
$dbname = "firefighter";
$username = "root";
$password = "";
$conn = mysqli_connect($servername, $username, $password, $dbname);

$query = mysqli_query(
    $conn,
    "SELECT * FROM reading_value ORDER BY id DESC"
);

$fetch_data = mysqli_fetch_array($query);
$class = $fetch_data["class"];

if ($class == "") {
    $class = "Tidak ada data class";
}

echo $class;
