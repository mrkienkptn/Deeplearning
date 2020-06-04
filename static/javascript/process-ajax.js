$(function () {
	$("#process-form").submit(function (e) {
		console.log("SONSONS")
		e.preventDefault()
		let link = $(this).attr('action')
		let type = $(".type:checked").val()
		console.log(link)

		$.ajax({
			type: 'POST',
			url : link,
			contentType: 'application/json;charset=UTF-8',
			data: {'im_type': type},
			success: data => {console.log(data)}
		})
		// sendreq.done(data => {
		// 	console.log(data.status)
		// 	$("#status").text(data.status)
		// })

	})
})