# Sem-6-PCAP

<h2>Edge detection algorithms:</h2>
<ol>
  <h3><li>First order derivative:</h3>
    <ul>
      <li>Sobel operator
      <li>Prewitt operator
      <li>Roberts operator
      <li>Scharr operator
    </ul>
  </li>
  <h3><li>Second order derivative:</h3></li>
    <ul>
      <li>Laplacian
      <li>Laplacian of Gaussian (LoG)
    </ul>
  </li>
  <h3><li>Canny operator</li></h3>
</ol>

<h3>Execution times(in seconds):</h3>
### Sobel Operator
|Image| GPU |CPU|
| -------- | ------- | ------- |
|pcapimg1|0.005|0.006|
|pcapimg2|0.219|0.312|
|pcapimg3|0.058|0.112|
### Prewitt Operator
|Image| GPU |CPU|
| -------- | ------- | ------- |
|pcapimg1|0.004|0.008|
|pcapimg2|0.206|0.315|
|pcapimg3|0.055|0.113|
### Roberts Operator
|Image| GPU |CPU|
| -------- | ------- | ------- |
|pcapimg1|0.006|0.005|
|pcapimg2|0.221|0.272|
|pcapimg3|0.061|0.074|
### Scharr Operator
|Image| GPU |CPU|
| -------- | ------- | ------- |
|pcapimg1|0.005|0.007|
|pcapimg2|0.247|0.444|
|pcapimg3|0.067|0.122|
### Laplacian
|Image| GPU |CPU|
| -------- | ------- | ------- |
|pcapimg1|0.004|0.006|
|pcapimg2|0.265|0.418|
|pcapimg3|0.065|0.131|
### Laplacian of Gaussian
|Image| GPU |CPU|
| -------- | ------- | ------- |
|pcapimg1|0.006|0.009|
|pcapimg2|0.303|0.464|
|pcapimg3|0.078|0.124|
### Canny Operator
|Image| GPU |CPU|
| -------- | ------- | ------- |
|pcapimg1|0.004|0.008|
|pcapimg2|0.184|0.469|
|pcapimg3|0.049|0.148|
