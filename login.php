<?php
session_start();
require_once "config.php";
require "navbarHome.php";

if ($_SERVER['REQUEST_METHOD'] == "POST") {

    $emailId = mysqli_real_escape_string($conn, $_POST['emailId']);
    $password = mysqli_real_escape_string($conn, $_POST['password']);


    $sql = "SELECT * FROM users WHERE emailId='$emailId' AND password='$password'";
    $result = mysqli_query($conn, $sql);


    if ($result && $result->num_rows == 1) {
        $_SESSION["loggedin"] = 1;
        $_SESSION['emailId'] = $emailId;
        $row = mysqli_fetch_assoc($result);
        $category = $row['category'];
        $loc = 'plantUser/dashboard.php';
        if ($category == 'Seller') {
            $loc = 'seller/sellerDashboard.php';
        }

        echo "<script>alert('LOGGED IN  successfully'); 
        window.location.href='$loc';</script>";
    } else {
        echo "<script>alert('INVALID credentials'); window.location.href='login.php';</script>";
    }
}
?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-4bw+/aepP/YC94hEpVNVgiZdgIC5+VKNBQNGCHeKRQN+PtmoHDEXuppvnDJzQIu9" crossorigin="anonymous">
    <script src="https://kit.fontawesome.com/3df97b829c.js" crossorigin="anonymous"></script>
    <title>LOGIN</title>
    <link rel="stylesheet" href="css/login.css">
</head>

<body>



    <div class="container">
        <h2>LOGIN</h2>
        <form action="" method="post">


            <div class="form-group">
                <label for="emailId">Email Id</label>
                <input type="text" class="form-control" id="emailId" name="emailId" aria-describedby="emailHelp" placeholder="Enter email-id" required>

            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" class="form-control" id="password" name="password" placeholder="Password" required>
            </div>
            <p>New User ? <a href="register.php">Register here</a></p>
            <div class="form-group text-center">
                <button type="submit" value="Login" class="btn btn-primary">Login</button>
            </div>


        </form>
    </div>
</body>