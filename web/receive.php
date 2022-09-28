<?php
$servername = "localhost";
$dbname = "firefighter";
$username = "root";
$password = "";


$score = $class_name = "";
echo var_dump($_POST);
$score      = test_input($_POST["score"]);
$class_name = test_input($_POST["class_name"]);

$conn = new mysqli($servername, $username, $password, $dbname, 3306);
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
$sql = "INSERT INTO reading_value (
    score, 
    class
    ) VALUES (
        '" . $score . "',
        '" . $class_name . "'
        )";

if ($conn->query($sql) === TRUE) {
    echo "New record created successfully";
} else {
    echo "Error: " . $sql . "<br>" . $conn->error;
}

$conn->close();



function test_input($data)
{
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);
    return $data;
}
?>