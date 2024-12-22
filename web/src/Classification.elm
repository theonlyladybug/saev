module Classification exposing (..)

import Browser
import Gradio
import Html
import Html.Attributes
import Html.Events
import Http
import Json.Decode as D
import Json.Encode as E
import Set


main =
    Browser.document
        { init = init
        , view = view
        , update = update
        , subscriptions = \model -> Sub.none
        }



-- MESSAGE


type Msg
    = HoverPatch Int
    | ResetHoveredPatch
    | ToggleSelectedPatch Int
    | GotImageUrl (Result Gradio.Error (Maybe String))



-- | GotSaeExamples (Result Gradio.Error (List SaeExampleResult))
-- MODEL


type alias Model =
    { err : Maybe String
    , exampleIndex : Int
    , imageUrl : Maybe String
    , hoveredPatchIndex : Maybe Int
    , selectedPatchIndices : Set.Set Int

    -- Progression
    , nPatchesExplored : Int
    , nPatchResets : Int
    , nImagesExplored : Int
    }


init : () -> ( Model, Cmd Msg )
init _ =
    let
        example =
            0
    in
    ( { err = Nothing
      , exampleIndex = example
      , imageUrl = Nothing
      , hoveredPatchIndex = Nothing
      , selectedPatchIndices = Set.empty

      -- Progression
      , nPatchesExplored = 0
      , nPatchResets = 0
      , nImagesExplored = 0
      }
    , getImageUrl example
      -- , Cmd.batch
      --     [ getImageUrl example, getPreds example ]
    )



-- UPDATE


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        HoverPatch i ->
            ( { model | hoveredPatchIndex = Just i }, Cmd.none )

        ResetHoveredPatch ->
            ( { model | hoveredPatchIndex = Nothing }, Cmd.none )

        GotImageUrl result ->
            case result of
                Ok url ->
                    ( { model | imageUrl = url }, Cmd.none )

                Err err ->
                    ( { model | imageUrl = Nothing, err = Just (explainGradioError err) }, Cmd.none )

        ToggleSelectedPatch i ->
            let
                ( patchIndices, nPatchesExplored, nPatchResets ) =
                    if Set.member i model.selectedPatchIndices then
                        ( Set.remove i model.selectedPatchIndices
                        , model.nPatchesExplored
                        , if Set.size model.selectedPatchIndices == 1 then
                            model.nPatchResets + 1

                          else
                            model.nPatchResets
                        )

                    else
                        ( Set.insert i model.selectedPatchIndices
                        , model.nPatchesExplored + 1
                        , model.nPatchResets
                        )
            in
            ( { model
                | selectedPatchIndices = patchIndices
                , nPatchesExplored = nPatchesExplored
                , nPatchResets = nPatchResets
              }
              -- , getSaeExamples model.exampleIndex patchIndices
            , Cmd.none
            )



-- API


explainGradioError : Gradio.Error -> String
explainGradioError err =
    case err of
        Gradio.NetworkError msg ->
            "Network error: " ++ msg

        Gradio.JsonError msg ->
            "Error decoding JSON: " ++ msg

        Gradio.ParsingError msg ->
            "Error parsing API response: " ++ msg

        Gradio.ApiError msg ->
            "Error in the API: " ++ msg


getImageUrl : Int -> Cmd Msg
getImageUrl img =
    Gradio.get "get-image"
        [ E.int img ]
        (D.list imgUrlDecoder |> D.map List.head)
        GotImageUrl



-- getSaeExamples : Int -> Set.Set Int -> Cmd Msg
-- getSaeExamples img patches =
--     Gradio.get
--         "get-sae-examples"
--         [ E.int img
--         , Set.toList patches |> E.list E.int
--         ]
--         (D.list saeExampleResultDecoder)
--         GotSaeExamples


type SaeExampleResult
    = SaeExampleMissing
    | SaeExampleUrl String
    | SaeExampleLatent Int


saeExampleResultDecoder : D.Decoder SaeExampleResult
saeExampleResultDecoder =
    D.oneOf
        [ imgUrlDecoder |> D.map SaeExampleUrl
        , D.int |> D.map SaeExampleLatent
        , D.null SaeExampleMissing
        ]


imgUrlDecoder : D.Decoder String
imgUrlDecoder =
    D.field "url" D.string
        |> D.map (String.replace "gradio_api/gradio_api" "gradio_api")
        |> D.map (String.replace "gra/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_ap/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_api/ca/gradio_api" "gradio_api")



-- VIEW


view : Model -> Browser.Document Msg
view model =
    { title = "Semantic Segmentation"
    , body =
        [ Html.header [] []
        , Html.main_
            []
            [ Html.div
                [ Html.Attributes.class "flex flex-row items-center" ]
                [ viewGriddedImage model.hoveredPatchIndex model.selectedPatchIndices model.imageUrl "Input Image"
                ]
            , viewErr model.err
            ]
        ]
    }


viewGriddedImage : Maybe Int -> Set.Set Int -> Maybe String -> String -> Html.Html Msg
viewGriddedImage hovered selected maybeUrl caption =
    case maybeUrl of
        Nothing ->
            Html.div [] [ Html.text "Loading..." ]

        Just url ->
            Html.div []
                [ Html.div
                    [ Html.Attributes.class "relative inline-block" ]
                    [ Html.div
                        [ Html.Attributes.class "absolute grid grid-rows-[repeat(16,_14px)] grid-cols-[repeat(16,_14px)] md:grid-rows-[repeat(16,_21px)] md:grid-cols-[repeat(16,_21px)]" ]
                        (List.map
                            (viewGridCell hovered selected)
                            (List.range 0 255)
                        )
                    , Html.img
                        [ Html.Attributes.class "block w-[224px] h-[224px] md:w-[336px] md:h-[336px]"
                        , Html.Attributes.src url
                        ]
                        []
                    ]
                , Html.p
                    []
                    [ Html.text caption ]
                ]


viewGridCell : Maybe Int -> Set.Set Int -> Int -> Html.Html Msg
viewGridCell hovered selected self =
    let
        classes =
            (case hovered of
                Just h ->
                    if h == self then
                        [ "border-2 border-rose-600 border-dashed" ]

                    else
                        []

                Nothing ->
                    []
            )
                ++ (if Set.member self selected then
                        [ "bg-rose-600/50" ]

                    else
                        []
                   )
    in
    Html.div
        ([ Html.Attributes.class "w-[14px] h-[14px] md:w-[21px] md:h-[21px]"
         , Html.Events.onMouseEnter (HoverPatch self)
         , Html.Events.onMouseLeave ResetHoveredPatch
         , Html.Events.onClick (ToggleSelectedPatch self)
         ]
            ++ List.map Html.Attributes.class classes
        )
        []


viewErr : Maybe String -> Html.Html Msg
viewErr err =
    case err of
        Just msg ->
            Html.p [] [ Html.text msg ]

        Nothing ->
            Html.span [] []
