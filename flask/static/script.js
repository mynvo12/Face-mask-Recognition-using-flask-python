$(document).scannerDetection({
	timeBeforeScanTest: 200, // wait for the next character for upto 200ms
	avgTimeByChar: 100, // it's not a barcode if a character takes longer than 100ms
	onComplete: function(barcode, qty){
    $('#pTest').text(barcode);    
    alert(barcode);
    } // main callback function	
});


$('#some-test-element').click(function() {
     $('div#focus-here').focus();  
})

//$('.search').find('input#foo').focus(); 