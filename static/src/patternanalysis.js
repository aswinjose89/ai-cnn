$( function() {
  $.widget( "custom.iconselectmenu", $.ui.selectmenu, {
    _renderItem: function( ul, item ) {
      var li = $( "<li>" ),
        wrapper = $( "<div>", { text: item.label } );

      if ( item.disabled ) {
        li.addClass( "ui-state-disabled" );
      }

      $( "<span>", {
        style: item.element.attr( "data-style" ),
        "class": "ui-icon " + item.element.attr( "data-class" )
      })
        .appendTo( wrapper );

      return li.append( wrapper ).appendTo( ul );
    }
  });

  $( "#sample_img" )
    .iconselectmenu()
    .iconselectmenu( "menuWidget")
      .addClass( "ui-menu-icons avatar" );
} );

var demo2 = $('.demo2').bootstrapDualListbox({
          preserveSelectionOnMove: 'moved',
          moveOnSelect: false,
          infoTextFiltered:'<span class="badge badge-warning">Filtered</span> {0} from {1}',
        });
