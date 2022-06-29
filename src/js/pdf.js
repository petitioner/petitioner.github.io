
	document.querySelector(".embed-link").addEventListener("click", function (e){
		e.preventDefault();
		this.setAttribute("class", "hidden");
		var options = {
			pdfOpenParams: {
				pagemode: "thumbs",
				navpanes: 0,
				toolbar: 0,
				statusbar: 0,
				view: "FitV"
			}
		};
		var myPDF = PDFObject.embed("../files/CV.pdf", "#pdf", options);
		var el = document.querySelector("#results");
		el.setAttribute("class", (myPDF) ? "success" : "fail");
		el.innerHTML = (myPDF) ? "PDFObject successfully added an &lt;embed> element to the page!" : "Uh-oh, the embed didn't work.";
	});
