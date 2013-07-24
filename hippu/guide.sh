#! /bin/bash

cat >guide.html <<HEADER
<!DOCTYPE html>
<html>
  <head>
    <title>aaltoasr-rec / aaltoasr-align User's Guide</title>
    <link href="guide.css" rel="stylesheet" type="text/css">
  </head>
  <body>
HEADER

markdown guide.md >>guide.html

cat >>guide.html <<FOOTER
  </body>
</html>
FOOTER
