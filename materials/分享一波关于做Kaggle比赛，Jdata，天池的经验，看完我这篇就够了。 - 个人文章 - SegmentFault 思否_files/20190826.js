/**
 * Created by sf on 2018/8/3.
 */
var areaArr = []

var trackHost = 'https://segmentfault.com';
var viewApi = trackHost + '/ad/track/view'
var clickApi = trackHost + '/ad/track/click'

function ajax(opt) {
    opt = opt || {};//opt以数组方式存参，如果参数不存在就为空。
    opt.method = opt.method.toUpperCase() || 'POST';//转为大写失败，就转为POST
    opt.url = opt.url || '';//检查URL是否为空
    opt.async = opt.async || true;//是否异步
    opt.data = opt.data || null;//是否发送参数，如POST、GET发送参数
    opt.success = opt.success || function () {}; //成功后处理方式
    var xmlHttp = null;//定义1个空变量
    if (XMLHttpRequest) {
        xmlHttp = new XMLHttpRequest();//如果XMLHttpRequest存在就新建，IE大于9&&非IE有效
    }
    else {
        xmlHttp = new ActiveXObject('Microsoft.XMLHTTP');//用于低版本IE
    }
    var params = [];//定义1个空数组
    for (var key in opt.data){
        params.push(key + '=' + opt.data[key]);//将opt里的data存到push里
    }
    var postData = params.join('&');//追加个& params
    if (opt.method.toUpperCase() === 'POST') {
        xmlHttp.open(opt.method, opt.url, opt.async);//开始请求
        xmlHttp.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded;charset=utf-8');//发送头信息，与表单里的一样。
        xmlHttp.send(postData);//发送POST数据
    }
    else if (opt.method.toUpperCase() === 'GET') {
        xmlHttp.open(opt.method, opt.url, opt.async);//GET请求
        xmlHttp.send(null);//发送空数据
    }
    xmlHttp.onreadystatechange = function () {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {//readyState有5个状态，可以百度一下看看都有什么。当请求完成，并且返回数据成功
            opt.success(xmlHttp.responseText);//返回成功的数据
        }
    };
}

var sTitle="";
window.SFGridAd = {};
SFGridAd.d = function(o) {
    sTitle = o.getAttribute('stitle');
    document.getElementById("gridMapHoverBox").style.display = "block"
}

SFGridAd.e = function(o) {
    sTitle = "";
    document.getElementById("gridMapHoverBox").style.display = "none"
}

SFGridAd.c = function(id) {

    var clickApi2 = clickApi + '?id=' + id

    ajax({url: clickApi2, method: 'GET'})
};

// 这里 data 需要注入
[{"id":"1750000020088766","user_id":"1030000002496563","box_ad_id":"0","started":"1565971200","ended":"1567180800","cycles":"2","status":"0","banner":"\/593\/351\/593351944-5d55f6f310553","link":"https:\/\/www.fundebug.com\/?utm_source=sf_ad","title":"\u4e00\u884c\u4ee3\u7801\u641e\u5b9aBUG\u76d1\u63a7","area_info":{"area":"F5:I5","height":"17","width":"68","left":"85","top":"68","pos":{"rowTop":5,"rowBottom":5,"columnLeft":6,"columnRight":9},"size":4},"created":"1565914824","modified":"1565914940"},{"id":"1750000020094983","user_id":"1030000007747454","box_ad_id":"0","started":"1566057600","ended":"1567267200","cycles":"1","status":"0","banner":"\/333\/830\/3338301951-5d565a184458c","link":"https:\/\/www.openinstall.io","title":"Android\/iOS \u4e00\u4e2a\u5305\u8d70\u5929\u4e0b \uff01","area_info":{"area":"F3:I3","height":"17","width":"68","left":"85","top":"34","pos":{"rowTop":3,"rowBottom":3,"columnLeft":6,"columnRight":9},"size":4},"created":"1565940198","modified":"1565940321"}].forEach(function(item) {
    var html = '<area shape="rect" ' +
        'target="_blank" ' +
        'onmouseover="SFGridAd.d(this)" ' +
        'onmouseout="SFGridAd.e(this)" ' +
        'onclick="SFGridAd.c(' + item.id + ')" ' +
        'coords="' + item.area_info.left + ',' + item.area_info.top + ',' + ((+item.area_info.left)+(+item.area_info.width)) + ',' + ((+item.area_info.top)+(+item.area_info.height)) + '" ' +
        'href="' + item.link + '" ' +
        'stitle="' + item.title + '" />'

    areaArr.push(html)
})

document.getElementById('gridsMap').innerHTML = areaArr.join('')

document.getElementById('gridsMap').onmousemove = function(e) {
    var pos = getMousePos(e)

    document.getElementById("gridMapHoverBox").style.left = pos.x + 20 + 'px'
    document.getElementById("gridMapHoverBox").style.top = pos.y + 20 + 'px'

    document.getElementById("gridMapHoverBox").innerHTML = sTitle
}

// 增加 view 统计
setTimeout(function() {
    isShow = document.querySelector('img[src^="https://static.segmentfault.com/sponsor"]').clientHeight > 0
    if (isShow) ajax({url: viewApi, method: 'GET'})
}, 0)

function getMousePos(event) {
    var e = event || window.event;
    var scrollX = document.documentElement.scrollLeft || document.body.scrollLeft;
    var scrollY = document.documentElement.scrollTop || document.body.scrollTop;
    var x = e.pageX || e.clientX + scrollX;
    var y = e.pageY || e.clientY + scrollY;
    return { 'x': x, 'y': y };
}
