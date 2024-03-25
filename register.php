<?php
require_once "config.php";
require "navbarHome.php";

if ($_SERVER['REQUEST_METHOD'] == "POST") {

    $firstName = mysqli_real_escape_string($conn, $_POST['firstName']);
    $lastName = mysqli_real_escape_string($conn, $_POST['lastName']);
    // $username = mysqli_real_escape_string($conn, $_POST['username']);
    $emailId = mysqli_real_escape_string($conn, $_POST['emailId']);
    $password = mysqli_real_escape_string($conn, $_POST['password']);
    $category = mysqli_real_escape_string($conn, $_POST['category']);
    // $emailParts = explode('@', $emailId);
    // $userId = $emailParts[0];
    // $address = NULL;

    $query1 = "SELECT emailId FROM users WHERE emailId = '$emailId'";
    $result1 = mysqli_query($conn, $query1);
    if ($result1){
        $rowCount = mysqli_num_rows($result1);
        if ($rowCount==0){
            $query2 = "INSERT INTO users VALUES('$emailId','$firstName','$lastName','$password','$category',NULL)";
            $result2 = mysqli_query($conn, $query2);
            if ($result2) {
    
                echo "<script>
                    alert('SIGN UP SUCCESSFUL YOU CAN LOGIN NOW');
                    window.location.href='index.php';
                 </script>";
            } else {
                echo "<center> <h3>user already exists!!</h3></center>" ;

            }

        }
        else{
            echo "<center> <h3>user already exists!!</h3></center>" ;
            
        }

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
    <title>REGISTER</title>
    <link rel="stylesheet" href="css/register.css">
</head>

<body>

    
    <div class="container" mt-4>
        <h2>REGISTER</h2>
        <!-- input form -->
        <form action="" method="POST">
            <div class="row">
                <div class="col">
                    <label for="firstName">First Name</label>
                    <input type="text" class="form-control" id="firstName" name="firstName" placeholder="First name" required>
                </div>
                <div class="col">
                    <label for="lastName">Last Name</label>
                    <input type="text" class="form-control" id="lastName" name="lastName" placeholder="Last name">
                </div>
            </div>


            <div class="form-group">
                <label for="emailId">Email address</label>
                <input type="email" class="form-control" id="emailId" name="emailId" aria-describedby="emailHelp" placeholder="Enter email" required>
                <small id="emailHelp" class="form-text text-muted">We'll never share your email with anyone else.</small>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" class="form-control" id="password" name="password" placeholder="Password" required>
            </div>
            <div class="form-group">
                <label for="confirmPassword">Confirm Password</label>
                <input type="password" class="form-control" id="confirmPassword" name="confirmPassword" placeholder="Re-enter Password" required oninput="check(this)">
            </div>

            <!-- select category -->


            <div class="form-row align-items-center">
                <div class="col-auto my-1">
                    <label class="mr-sm-2" for="category">Category</label>
                    <select class="custom-select mr-sm-2" id="category" name="category" required>
                        <option selected disabled hidden>select</option>
                        <option value="Plant Enthusiast">Plant Enthusiast</option>
                        <option value="Seller">Seller</option>

                    </select>
                </div>

            </div>

            <!-- select category code ends -->


            <button type="submit" class="btn btn-primary">Register</button>
            <p>Already a user? <a href="login.php">Login</a></p>
        <script>
            function check(input) {
                if (input.value != document.getElementById('password').value) {
                    input.setCustomValidity('Passwords must match.');
                } else {
                    input.setCustomValidity('');
                }
            }
        </script>
        </form>

    </div>
</body>